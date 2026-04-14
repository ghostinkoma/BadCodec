/**
 * @file  main.c
 * @brief BadCodecPlayer v1.0.0
 *
 * 変更点:
 *   - フレームレートを .bad ヘッダ値 (CFG_TARGET_FPS==0) または
 *     CFG_TARGET_FPS で指定した fps から自動計算
 *   - adpcm_set_frame_ms() で音声同期タイマに通知
 *   - OSD (CPU/FPS/VU) を frame_osd_update() で描画
 *   - CPU 使用率を esp_cpu_load で取得
 */

 #include <string.h>
 #include "freertos/FreeRTOS.h"
 #include "freertos/task.h"
 #include "freertos/timers.h"
 #include "freertos/semphr.h"
 #include "esp_log.h"
 #include "esp_timer.h"
 #include "esp_freertos_hooks.h"

 #include "config.h"
 #include "ssd1306_drv.h"
 #include "bad_decode.h"
 #include "bad_data.h"
 #include "adpcm_drv.h"
 #include "draw.h"
 
 static const char *TAG = "BadCodec";
 
 static SemaphoreHandle_t s_video_sem;
 
 static uint8_t   s_gram[BAD_GRAM_SIZE(CFG_VIDEO_W, CFG_VIDEO_H)];
 static uint8_t   s_prev[BAD_GRAM_SIZE(CFG_VIDEO_W, CFG_VIDEO_H)];
 static bad_ctx_t s_ctx;
 
 static TimerHandle_t s_frame_timer;
 
 /* OSD 計測用 */
 static volatile uint32_t s_frame_count  = 0;  /* 前回計測からのフレーム数 */
 static volatile uint32_t s_fps_x10      = 0;  /* 実測 FPS × 10 */
 static volatile uint32_t s_cpu_x10      = 0;  /* CPU 使用率 × 10 */
 
 /* フレーム間隔 (ms) — bad ヘッダまたは CFG_TARGET_FPS から決定 */
 static uint32_t s_frame_ms = 29U;
 
 /* ---- ソフトタイマコールバック ---- */
 static void frame_timer_cb(TimerHandle_t xTimer)
 {
     (void)xTimer;
     xSemaphoreGive(s_video_sem);
 }
 
 static void start_sw_frame_timer(void)
 {
     s_frame_timer = xTimerCreate("frm",
                                  pdMS_TO_TICKS(s_frame_ms),
                                  pdTRUE, NULL, frame_timer_cb);
     if (!s_frame_timer) { ESP_LOGE(TAG,"xTimerCreate failed"); esp_restart(); }
     xTimerStart(s_frame_timer, portMAX_DELAY);
     ESP_LOGI(TAG,"SW timer: %"PRIu32"ms/frame (%.1f fps)",
              s_frame_ms, 1000.0f / (float)s_frame_ms);
 }
 
 /* ---- read callbacks ---- */
 #ifdef CFG_SOURCE_FLASH
 static uint16_t flash_read(bad_addr_t off, uint8_t *buf, uint16_t len)
 {
     memcpy(buf, bad_data + off, len);
     return len;
 }
 #endif
 
 #ifdef CFG_SOURCE_SD
 #include "esp_vfs_fat.h"
 #include "driver/sdmmc_host.h"
 #include "sdmmc_cmd.h"
 #include <stdio.h>
 static FILE *s_sd_fp = NULL;
 static long  s_sd_last = -1L;
 static uint16_t sd_read(bad_addr_t off, uint8_t *buf, uint16_t len)
 {
     if (!s_sd_fp) return 0;
     if ((long)off != s_sd_last) fseek(s_sd_fp, (long)off, SEEK_SET);
     size_t n = fread(buf, 1, len, s_sd_fp);
     s_sd_last = (long)off + (long)n;
     return (uint16_t)n;
 }
 static bool sd_mount(void)
 {
     esp_vfs_fat_sdmmc_mount_config_t mc = {
         .format_if_mount_failed=false,.max_files=2,.allocation_unit_size=16*1024};
     sdmmc_card_t *card;
     sdmmc_host_t host = SDMMC_HOST_DEFAULT();
     sdmmc_slot_config_t slot = SDMMC_SLOT_CONFIG_DEFAULT();
     slot.width = 1;
     if (esp_vfs_fat_sdmmc_mount("/sdcard",&host,&slot,&mc,&card)!=ESP_OK) return false;
     s_sd_fp = fopen(CFG_SD_FILE,"rb");
     return s_sd_fp != NULL;
 }
 #endif
 
 static void show_error(void)
 {
     for (;;) {
         memset(g_fb,0xFF,sizeof(g_fb)); ssd1306_flush();
         vTaskDelay(pdMS_TO_TICKS(400));
         ssd1306_clear(); ssd1306_flush();
         vTaskDelay(pdMS_TO_TICKS(400));
     }
 }
 
 /* ============================================================
  * FPS / CPU 計測
  *
  * CPU使用率計測方式: esp_register_freertos_idle_hook()
  *
  * ESP-IDF v5.0.x はアイドルフックを独自のフックチェーンで管理する。
  * vApplicationIdleHook() への直接書き込みは二重定義でクラッシュする。
  * esp_register_freertos_idle_hook() を使えば安全に追加できる。
  * sdkconfig の変更も不要。
  *
  * 計測原理:
  *   アイドルフックが呼ばれるたびに s_idle_count++ する。
  *   stats_task が 1 秒ごとに増分を計測し、
  *   最初の 1 秒間(最もアイドルが多いと仮定)を基準値として
  *   CPU 使用率を算出する。
  *
  *   初回計測から 2 秒目以降に正しい値が表示される。
  * ============================================================ */
 static volatile uint32_t s_idle_count = 0;

 static bool IRAM_ATTR idle_hook_cb(void)
 {
     s_idle_count++;
     return false;  /* false = スリープしない (音声ISRのため必須) */
 }

 static void stats_task(void *arg)
 {
     (void)arg;

     /* アイドルフックを登録 */
     esp_register_freertos_idle_hook_for_cpu(idle_hook_cb, 0);

     uint32_t last_fc    = 0;
     uint32_t last_idle  = 0;
     int64_t  last_us    = esp_timer_get_time();
     static uint32_t s_idle_baseline = 0;

     for (;;) {
         vTaskDelay(pdMS_TO_TICKS(1000));

         int64_t  now_us   = esp_timer_get_time();
         uint32_t fc       = s_frame_count;
         uint32_t idle_now = s_idle_count;

         /* FPS × 10 */
         uint32_t delta_fc = fc - last_fc;
         int64_t  dt_us    = now_us - last_us;
         s_fps_x10 = (dt_us > 0)
                     ? (uint32_t)((int64_t)delta_fc * 10000000LL / dt_us)
                     : 0;

         /* CPU 使用率 × 10
          * 初回の d_idle を「100% アイドル時の基準値」として記録。
          * 以降: CPU使用率 = (1 - d_idle/baseline) × 100              */
         uint32_t d_idle = idle_now - last_idle;
         if (s_idle_baseline == 0U && d_idle > 0U) {
             s_idle_baseline = d_idle;   /* 初回: 基準値を記録 */
             s_cpu_x10 = 0;
         } else if (s_idle_baseline > 0U) {
             if (d_idle >= s_idle_baseline) {
                 s_cpu_x10 = 0;
             } else {
                 s_cpu_x10 = (s_idle_baseline - d_idle) * 1000U
                             / s_idle_baseline;
             }
         } else {
             s_cpu_x10 = 0;
         }

         last_fc   = fc;
         last_idle = idle_now;
         last_us   = now_us;
     }
 }
 
 /* ============================================================
  * player_task
  * ============================================================ */
 static void player_task(void *arg)
 {
     (void)arg;
 
     frame_t *frame = frame_create();
 
 #if (CFG_VIDEO_ENABLE == 1)
 #ifdef CFG_SOURCE_FLASH
     s_ctx.read = flash_read;
 #endif
 #ifdef CFG_SOURCE_SD
     if (!sd_mount()) { ESP_LOGE(TAG,"SD mount failed"); show_error(); }
     s_ctx.read = sd_read;
 #endif
     s_ctx.gram     = s_gram;
     s_ctx.prev     = s_prev;
     s_ctx.buf_size = (uint16_t)sizeof(s_gram);
 
     if (bad_init(&s_ctx) != BAD_OK) {
         ESP_LOGE(TAG,"bad_init failed"); show_error();
     }
 
     /* ---- フレームレート決定 --------------------------------
      * CFG_TARGET_FPS == 0: .bad ヘッダから取得
      * CFG_TARGET_FPS != 0: 指定値を使用
      * -------------------------------------------------------- */
     uint32_t fps = 0;
 #if (CFG_TARGET_FPS != 0)
     fps = (uint32_t)CFG_TARGET_FPS;
 #else
     /* bad_ctx_t に fps フィールドがある場合は使用。
      * ない場合はデフォルト 34fps (29ms) を使用する。
      * (bad_decode.h に total_frames はあるが fps は版依存)    */
     if (s_ctx.total_frames > 0) {
         /* fps フィールドが存在すれば ctx.fps を使う。
          * 現バージョンの bad_decode.h には fps がないため
          * デフォルト 34fps を使用。必要に応じて追加してください。 */
         fps = 34U;  /* デフォルト: 34fps ≈ 29ms/frame */
     }
 #endif
     if (fps == 0) fps = 34U;
     s_frame_ms = 1000U / fps;
     if (s_frame_ms == 0) s_frame_ms = 1;
 
     ESP_LOGI(TAG,"%ux%u  %u frames  fps=%"PRIu32"  frame_ms=%"PRIu32,
              s_ctx.width, s_ctx.height,
              s_ctx.total_frames, fps, s_frame_ms);
 
     /* 音声ドライバにフレーム間隔を通知 */
 #if (CFG_AUDIO_ENABLE == 1)
     adpcm_set_frame_ms(s_frame_ms);
 #endif
 
 #else
     ESP_LOGW(TAG,"VIDEO disabled");
 #endif /* CFG_VIDEO_ENABLE */
 
     /* ---- メインループ ---- */
     for (;;) {
         xSemaphoreTake(s_video_sem, portMAX_DELAY);
 
 #if (CFG_VIDEO_ENABLE == 1)
         bad_result_t r = bad_next_frame(&s_ctx);
 
         if (r == BAD_OK || r == BAD_EOF) {
             ssd1306_clear();
             ssd1306_blit_gram(s_ctx.gram, (int)s_ctx.width, (int)s_ctx.height);
 
             /* OSD 描画 */
 #if (CFG_OSD_CPU || CFG_OSD_FPS || CFG_OSD_VU)
             frame_osd_update(frame,
                              s_fps_x10,
                              s_cpu_x10,
                              adpcm_get_vu());
 #endif
             ssd1306_flush();
             s_frame_count++;
         }
 
         if (r == BAD_EOF) {
             bad_rewind(&s_ctx);
 #if (CFG_AUDIO_ENABLE == 1)
             adpcm_rewind();
 #endif
 #ifdef CFG_SOURCE_SD
             s_sd_last = -1L;
 #endif
         } else if (r != BAD_OK) {
             ESP_LOGW(TAG,"frame %u err %d, rewind", s_ctx.current_frame, (int)r);
             bad_rewind(&s_ctx);
 #if (CFG_AUDIO_ENABLE == 1)
             adpcm_rewind();
 #endif
         }
 #endif /* CFG_VIDEO_ENABLE */
     }
 }
 
 /* ============================================================
  * app_main
  * ============================================================ */
 void app_main(void)
 {
     ESP_LOGI(TAG,"BadCodecPlayer v1.0.0  VIDEO=%d AUDIO=%d",
              CFG_VIDEO_ENABLE, CFG_AUDIO_ENABLE);
 
     s_video_sem = xSemaphoreCreateBinary();
     if (!s_video_sem) { ESP_LOGE(TAG,"sem fail"); esp_restart(); }
 
 #if (CFG_VIDEO_ENABLE == 1)
     ssd1306_init();
     memset(g_fb,0xFF,sizeof(g_fb)); ssd1306_flush();
     vTaskDelay(pdMS_TO_TICKS(200));
     ssd1306_clear(); ssd1306_flush();
 #endif
 
 #if (CFG_AUDIO_ENABLE == 1)
     if (adpcm_init(NULL, 0, s_video_sem) == 0) {
         ESP_LOGI(TAG,"Audio: gptimer ISR");
     } else {
         ESP_LOGE(TAG,"adpcm_init failed: SW timer fallback");
         start_sw_frame_timer();
     }
 #else
     start_sw_frame_timer();
 #endif
 
     /* OSD/FPS 計測タスク */
 #if (CFG_OSD_CPU || CFG_OSD_FPS || CFG_OSD_VU)
     xTaskCreate(stats_task, "stats", 2048, NULL, 2, NULL);
 #endif
 
     xTaskCreate(player_task, "player", 6144, NULL, 5, NULL);
 }
 
