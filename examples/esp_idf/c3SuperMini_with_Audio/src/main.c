/**
 * @file  main.c
 * @brief BadCodecPlayer - ESP32-C3 Super Mini / SSD1306 128x64
 * @version v0.6.3  (Protocol 514, SPEC rev.19)
 *
 * Debug switches (config.h):
 *   CFG_VIDEO_ENABLE  1/0  映像の有効/無効
 *   CFG_AUDIO_ENABLE  1/0  音声の有効/無効
 *
 * 切り分け手順:
 *   Step1: VIDEO=1, AUDIO=0  映像のみ動作確認
 *   Step2: VIDEO=0, AUDIO=1  音声のみ動作確認（シリアルログ確認）
 *   Step3: VIDEO=1, AUDIO=1  両方同時（本来の動作）
 */

 #include <string.h>
 #include "freertos/FreeRTOS.h"
 #include "freertos/task.h"
 #include "freertos/timers.h"
 #include "freertos/semphr.h"
 #include "esp_log.h"
 
 #include "config.h"
 #include "ssd1306_drv.h"
 #include "bad_decode.h"
 #include "bad_data.h"
 #include "adpcm_drv.h"
 
 static const char *TAG = "BadCodec";
 
 /* ---- セマフォ: 音声ISR または ソフトウェアタイマが1フレームごとに give ---- */
 static SemaphoreHandle_t s_video_sem;
 
 /* ---- デコーダバッファ ---- */
 static uint8_t   s_gram[BAD_GRAM_SIZE(CFG_VIDEO_W, CFG_VIDEO_H)];
 static uint8_t   s_prev[BAD_GRAM_SIZE(CFG_VIDEO_W, CFG_VIDEO_H)];
 static bad_ctx_t s_ctx;
 
 /* ---- ソフトウェアフレームタイマ (AUDIO=0 時 / フォールバック時) ----
  *
  * 注意: FreeRTOS ソフトウェアタイマのコールバックはタイマサービスタスク
  * (デーモンタスク) のコンテキストで呼ばれる。ISR ではないため
  * xSemaphoreGive() (非ISR版) を使う。
  * FromISR 版を使うとスタック破壊・クラッシュの原因になる。
  * ------------------------------------------------------------------ */
 static TimerHandle_t s_frame_timer;
 
 static void frame_timer_cb(TimerHandle_t xTimer)
 {
     (void)xTimer;
     /* タスクコンテキスト: 非ISR版 xSemaphoreGive を使う */
     xSemaphoreGive(s_video_sem);
 }
 
 /* ---- ソフトウェアタイマ起動ヘルパ ---- */
 static void start_sw_frame_timer(void)
 {
     s_frame_timer = xTimerCreate(
         "frm",
         pdMS_TO_TICKS(CFG_FRAME_MS),
         pdTRUE,
         NULL,
         frame_timer_cb);
 
     if (s_frame_timer == NULL) {
         ESP_LOGE(TAG, "xTimerCreate failed");
         esp_restart();
     }
     xTimerStart(s_frame_timer, portMAX_DELAY);
     ESP_LOGI(TAG, "SW frame timer started: %ums/frame", (unsigned)CFG_FRAME_MS);
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
 static FILE *s_sd_fp   = NULL;
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
         .format_if_mount_failed = false,
         .max_files = 2,
         .allocation_unit_size = 16 * 1024,
     };
     sdmmc_card_t *card;
     sdmmc_host_t  host = SDMMC_HOST_DEFAULT();
     sdmmc_slot_config_t slot = SDMMC_SLOT_CONFIG_DEFAULT();
     slot.width = 1;
     if (esp_vfs_fat_sdmmc_mount("/sdcard", &host, &slot, &mc, &card) != ESP_OK)
         return false;
     s_sd_fp = fopen(CFG_SD_FILE, "rb");
     return s_sd_fp != NULL;
 }
 #endif /* CFG_SOURCE_SD */
 
 /* ---- エラー表示: 画面全体を点滅 ---- */
 static void show_error(void)
 {
     for (;;) {
         memset(g_fb, 0xFF, sizeof(g_fb));
         ssd1306_flush();
         vTaskDelay(pdMS_TO_TICKS(400));
         ssd1306_clear();
         ssd1306_flush();
         vTaskDelay(pdMS_TO_TICKS(400));
     }
 }
 
 /* ============================================================
  * player_task
  * ============================================================ */
 static void player_task(void *arg)
 {
     (void)arg;
 
 #if (CFG_VIDEO_ENABLE == 1)
 
 #ifdef CFG_SOURCE_FLASH
     s_ctx.read = flash_read;
     ESP_LOGI(TAG, "Source: Flash (bad_data[])");
 #endif
 #ifdef CFG_SOURCE_SD
     if (!sd_mount()) {
         ESP_LOGE(TAG, "SD mount failed");
         show_error();
     }
     s_ctx.read = sd_read;
     ESP_LOGI(TAG, "Source: SD %s", CFG_SD_FILE);
 #endif
 
     s_ctx.gram     = s_gram;
     s_ctx.prev     = s_prev;
     s_ctx.buf_size = (uint16_t)sizeof(s_gram);
 
     if (bad_init(&s_ctx) != BAD_OK) {
         ESP_LOGE(TAG, "bad_init failed");
         show_error();
     }
     ESP_LOGI(TAG, "%ux%u  %u frames  ~%ums/frame",
              s_ctx.width, s_ctx.height,
              s_ctx.total_frames, (unsigned)CFG_FRAME_MS);
 
 #else
     ESP_LOGW(TAG, "VIDEO disabled (CFG_VIDEO_ENABLE=0)");
 #endif /* CFG_VIDEO_ENABLE */
 
     /* ---- メインループ ---- */
     for (;;) {
         /* セマフォ待ち:
          *   AUDIO=1 → 音声ISRが ADPCM_SAMPLES_PER_FRAME ごとに give
          *   AUDIO=0 → ソフトウェアタイマが CFG_FRAME_MS ごとに give */
         xSemaphoreTake(s_video_sem, portMAX_DELAY);
 
 #if (CFG_VIDEO_ENABLE == 1)
         bad_result_t r = bad_next_frame(&s_ctx);
 
         if (r == BAD_OK || r == BAD_EOF) {
             ssd1306_clear();
             ssd1306_blit_gram(s_ctx.gram,
                               (int)s_ctx.width,
                               (int)s_ctx.height);
             ssd1306_flush();
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
             ESP_LOGW(TAG, "frame %u: decode err %d, rewind",
                      (unsigned)s_ctx.current_frame, (int)r);
             bad_rewind(&s_ctx);
 #if (CFG_AUDIO_ENABLE == 1)
             adpcm_rewind();
 #endif
 #ifdef CFG_SOURCE_SD
             s_sd_last = -1L;
 #endif
         }
 #endif /* CFG_VIDEO_ENABLE */
     }
 }
 
 /* ============================================================
  * app_main
  *
  * 初期化順序:
  *   1. セマフォ作成
  *   2. OLED 初期化 + 起動フラッシュ  (VIDEO=1 時)
  *   3. 音声初期化 OR ソフトタイマ起動 (フレームクロック源を確定)
  *   4. player_task 起動
  *
  * gptimer (音声ISR) は必ず player_task 起動直前に start する。
  * ============================================================ */
 void app_main(void)
 {
     ESP_LOGI(TAG, "BadCodecPlayer v0.6.3  VIDEO=%d AUDIO=%d",
              (int)CFG_VIDEO_ENABLE, (int)CFG_AUDIO_ENABLE);
 
     /* 1. セマフォ */
     s_video_sem = xSemaphoreCreateBinary();
     if (s_video_sem == NULL) {
         ESP_LOGE(TAG, "semaphore create failed");
         esp_restart();
     }
 
     /* 2. OLED */
 #if (CFG_VIDEO_ENABLE == 1)
     ssd1306_init();
     memset(g_fb, 0xFF, sizeof(g_fb));
     ssd1306_flush();
     vTaskDelay(pdMS_TO_TICKS(200));
     ssd1306_clear();
     ssd1306_flush();
 #else
     ESP_LOGW(TAG, "VIDEO disabled: OLED skipped");
 #endif
 
     /* 3. フレームクロック源 */
 #if (CFG_AUDIO_ENABLE == 1)
     if (adpcm_init(NULL, 0, s_video_sem) == 0) {
         ESP_LOGI(TAG, "Audio clock source: gptimer ISR");
     } else {
         ESP_LOGE(TAG, "adpcm_init failed: using SW timer fallback");
         start_sw_frame_timer();
     }
 #else
     start_sw_frame_timer();
 #endif
 
     /* 4. player_task */
     xTaskCreate(player_task, "player", 6144, NULL, 5, NULL);
 }
 
