/**
 * @file  main.c
 * @brief BadCodecPlayer v3.3.0
 *
 * v3.3.0 バグ修正:
 *   [1] PAUSE で音声も停止
 *       adpcm_set_frame_ms(0) で gptimer ISR の Give を停止。
 *       resume 時は adpcm_set_frame_us() で再開。
 *   [2] OSD マスクを正しく適用
 *       g_btn.osd_mask bit0=CPU bit1=FPS bit2=VU で
 *       frame_osd_update() への引数を動的に制御。
 *       毎フレーム osd_layer_clear() を呼ぶことで残像を防ぐ。
 *   [3] 音量調整を実機に反映
 *       button_get_vol() の戻り値を毎フレーム adpcm_set_vol() に渡す。
 *       vol_step=0 のとき 0 (無音) を渡す。
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
#include "button.h"

static const char *TAG = "BadCodec";

static SemaphoreHandle_t s_video_sem;
static uint8_t   s_gram[BAD_GRAM_SIZE(CFG_VIDEO_W, CFG_VIDEO_H)];
static uint8_t   s_prev[BAD_GRAM_SIZE(CFG_VIDEO_W, CFG_VIDEO_H)];
static bad_ctx_t s_ctx;
static TimerHandle_t s_frame_timer;

static volatile uint32_t s_frame_count = 0;
static volatile uint32_t s_fps_x10     = 0;
static volatile uint32_t s_cpu_x10     = 0;
static uint32_t s_frame_ms  = 33U;
static uint32_t s_frame_us  = 33333U;

/* ---- ソフトタイマ (音声無効時フォールバック) -------------- */
static void frame_timer_cb(TimerHandle_t t)
{
    (void)t; xSemaphoreGive(s_video_sem);
}
static void start_sw_frame_timer(void)
{
    s_frame_timer = xTimerCreate("frm", pdMS_TO_TICKS(s_frame_ms),
                                 pdTRUE, NULL, frame_timer_cb);
    if (!s_frame_timer) { esp_restart(); }
    xTimerStart(s_frame_timer, portMAX_DELAY);
}

/* ---- flash read ------------------------------------------ */
#ifdef CFG_SOURCE_FLASH
static uint16_t flash_read(bad_addr_t off, uint8_t *buf, uint16_t len)
{
    memcpy(buf, bad_data + off, len); return len;
}
#endif

#ifdef CFG_SOURCE_SD
#include "esp_vfs_fat.h"
#include "driver/sdmmc_host.h"
#include "sdmmc_cmd.h"
#include <stdio.h>
static FILE *s_sd_fp  = NULL;
static long  s_sd_last = -1L;
static uint16_t sd_read(bad_addr_t off, uint8_t *buf, uint16_t len)
{
    if (!s_sd_fp) { return 0; }
    if ((long)off != s_sd_last) { fseek(s_sd_fp,(long)off,SEEK_SET); }
    size_t n = fread(buf,1,len,s_sd_fp);
    s_sd_last=(long)off+(long)n;
    return (uint16_t)n;
}
static bool sd_mount(void)
{
    esp_vfs_fat_sdmmc_mount_config_t mc={
        .format_if_mount_failed=false,.max_files=2,
        .allocation_unit_size=16*1024};
    sdmmc_card_t *card;
    sdmmc_host_t host=SDMMC_HOST_DEFAULT();
    sdmmc_slot_config_t slot=SDMMC_SLOT_CONFIG_DEFAULT();
    slot.width=1;
    if(esp_vfs_fat_sdmmc_mount("/sdcard",&host,&slot,&mc,&card)!=ESP_OK){
        return false;
    }
    s_sd_fp=fopen(CFG_SD_FILE,"rb");
    return s_sd_fp!=NULL;
}
#endif

static void show_error(void)
{
    for(;;){
        memset(g_fb,0xFF,sizeof(g_fb)); ssd1306_flush();
        vTaskDelay(pdMS_TO_TICKS(400));
        ssd1306_clear(); ssd1306_flush();
        vTaskDelay(pdMS_TO_TICKS(400));
    }
}

/* ============================================================
 * stats_task
 * ============================================================ */
static volatile uint32_t s_idle_cnt = 0;
static bool IRAM_ATTR idle_hook_cb(void) { s_idle_cnt++; return false; }

static void stats_task(void *arg)
{
    (void)arg;
    esp_register_freertos_idle_hook_for_cpu(idle_hook_cb, 0);
    uint32_t last_fc=0, last_idle=0;
    int64_t  last_us=esp_timer_get_time();
    static uint32_t idle_base=0;
    for(;;){
        vTaskDelay(pdMS_TO_TICKS(1000));
        int64_t  now=esp_timer_get_time();
        uint32_t fc=s_frame_count, idle=s_idle_cnt;
        int64_t  dt=now-last_us;
        s_fps_x10 = (dt>0) ?
            (uint32_t)((int64_t)(fc-last_fc)*10000000LL/dt) : 0;
        uint32_t di=idle-last_idle;
        if(idle_base==0 && di>0){ idle_base=di; }
        s_cpu_x10 = (idle_base>0 && di<idle_base) ?
            (idle_base-di)*1000U/idle_base : 0;
        last_fc=fc; last_idle=idle; last_us=now;
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
    if (!sd_mount()) { show_error(); }
    s_ctx.read = sd_read;
#endif
    s_ctx.gram     = s_gram;
    s_ctx.prev     = s_prev;
    s_ctx.buf_size = (uint16_t)sizeof(s_gram);
    if (bad_init(&s_ctx) != BAD_OK) { show_error(); }

    /* ---- FPS 決定 ----------------------------------------- */
#if defined(CFG_TARGET_FPS_100) && (CFG_TARGET_FPS_100 != 0)
    uint32_t fps_100 = (uint32_t)CFG_TARGET_FPS_100;
#elif defined(CFG_TARGET_FPS) && (CFG_TARGET_FPS != 0)
    uint32_t fps_100 = (uint32_t)CFG_TARGET_FPS * 100U;
#else
    uint32_t fps_100 = 3000U;
#endif
    if (fps_100 == 0U) { fps_100 = 3000U; }

    s_frame_us = 100000000U / fps_100;
    if (s_frame_us == 0U) { s_frame_us = 1U; }
    s_frame_ms = s_frame_us / 1000U;
    if (s_frame_ms == 0U) { s_frame_ms = 1U; }

    ESP_LOGI(TAG,"%ux%u %u frames fps=%.2f frame_us=%"PRIu32,
             s_ctx.width, s_ctx.height, s_ctx.total_frames,
             (float)fps_100/100.0f, s_frame_us);

#if (CFG_AUDIO_ENABLE == 1)
    adpcm_set_frame_us(s_frame_us);
#endif
    if (s_frame_timer) {
        xTimerChangePeriod(s_frame_timer,
                           pdMS_TO_TICKS(s_frame_ms), portMAX_DELAY);
    }
#else
    ESP_LOGW(TAG,"VIDEO disabled");
#endif /* CFG_VIDEO_ENABLE */

    int was_paused = 0;

    for (;;) {

        /* ============================================================
         * [Fix1] PAUSE: 音声も停止する
         *   adpcm_set_frame_ms(0) → ISR の s_frame_interval_ms=0
         *   → spf=0 → Give が発生しない → セマフォが積まれない
         *   → xSemaphoreTake がブロック → 映像も停止
         *   音声 PCM の出力 (ledc_set_duty) は最後の値で保持
         *   (ポップノイズなし)
         * ============================================================ */
        if (g_btn.paused) {
            if (!was_paused) {
#if (CFG_AUDIO_ENABLE == 1)
                adpcm_pause();   /* gptimer 停止 → 音声完全停止 */
#endif
                /* 積み上がったセマフォを捨てる */
                while (xSemaphoreTake(s_video_sem, 0) == pdTRUE) {}
                was_paused = 1;
                ESP_LOGI(TAG,"PAUSE");
            }
            vTaskDelay(pdMS_TO_TICKS(100));
#if (CFG_VIDEO_ENABLE == 1)
            ssd1306_clear();
            ssd1306_blit_gram(s_ctx.gram,
                              (int)s_ctx.width, (int)s_ctx.height);
            osd_layer_clear();
            osd_draw_pause();
            osd_layer_blit();
            ssd1306_flush();
#endif
            continue;
        }

        if (was_paused) {
            was_paused = 0;
#if (CFG_AUDIO_ENABLE == 1)
            adpcm_resume();   /* gptimer 再開、カウンタリセット */
#endif
            while (xSemaphoreTake(s_video_sem, 0) == pdTRUE) {}
            ESP_LOGI(TAG,"RESUME");
        }

        /* ============================================================
         * [Fix3] 音量を毎フレーム反映
         *   button_get_vol() の戻り値 (0-256) を adpcm_set_vol() に渡す。
         *   ISR が s_vol を参照するので即時反映される。
         *   vol_step=0 のとき button_get_vol() は 0 を返す → 無音。
         * ============================================================ */
#if (CFG_AUDIO_ENABLE == 1)
        adpcm_set_vol(button_get_vol());
#endif

        xSemaphoreTake(s_video_sem, portMAX_DELAY);

#if (CFG_VIDEO_ENABLE == 1)
        /* フレームスキップ (最大1フレーム) */
        if (xSemaphoreTake(s_video_sem, 0) == pdTRUE) {
            bad_result_t rs = bad_next_frame(&s_ctx);
            if (rs == BAD_EOF) {
                bad_rewind(&s_ctx);
#if (CFG_AUDIO_ENABLE == 1)
                adpcm_rewind();
#endif
            }
        }

        bad_result_t r = bad_next_frame(&s_ctx);

        if (r == BAD_OK || r == BAD_EOF) {
            ssd1306_clear();
            ssd1306_blit_gram(s_ctx.gram,
                              (int)s_ctx.width, (int)s_ctx.height);

            /* ============================================================
             * [Fix2] OSD マスクを正しく適用
             *   毎フレーム osd_layer_clear() を呼ぶ → 残像なし。
             *   g_btn.osd_mask の各ビットで表示/非表示を制御:
             *     bit0=CPU: 0xFFFFFFFF を渡すと非表示
             *     bit1=FPS: 0xFFFFFFFF を渡すと非表示
             *     bit2=VU : 0 を渡すと非表示 (バー幅=0)
             *   osd_mask=0 のときは frame_osd_update 自体を呼ばない。
             * ============================================================ */
            osd_layer_clear();

            if (g_btn.osd_mask != 0U) {
                uint32_t show_fps = (g_btn.osd_mask & 0x02U)
                                    ? s_fps_x10 : 0xFFFFFFFFUL;
                uint32_t show_cpu = (g_btn.osd_mask & 0x01U)
                                    ? s_cpu_x10 : 0xFFFFFFFFUL;
                uint16_t show_vu  = (g_btn.osd_mask & 0x04U)
                                    ? adpcm_get_vu() : 0U;
                frame_osd_update(frame, show_fps, show_cpu, show_vu);
            }

            if (g_btn.vol_show) {
                osd_draw_vol(g_btn.vol_step);
            }

            osd_layer_blit();
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
            ESP_LOGW(TAG,"frame %u err %d",
                     s_ctx.current_frame, (int)r);
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
    ESP_LOGI(TAG,"BadCodecPlayer v3.3.0  VIDEO=%d AUDIO=%d",
             CFG_VIDEO_ENABLE, CFG_AUDIO_ENABLE);

    s_video_sem = xSemaphoreCreateBinary();
    if (!s_video_sem) { esp_restart(); }

    /* ---- ボタン初期化
     * OSD 初期マスク: config.h の CFG_OSD_CPU/FPS/VU から生成
     * 音量初期ステップ: CFG_AUDIO_VOL から変換           ---- */
    {
        uint8_t vs = (uint8_t)(((uint32_t)CFG_AUDIO_VOL * 16U) / 256U);
        if (vs >= 16U) { vs = 15U; }
        button_init(vs);
    }

#if (CFG_VIDEO_ENABLE == 1)
    ssd1306_init();
    memset(g_fb,0xFF,sizeof(g_fb)); ssd1306_flush();
    vTaskDelay(pdMS_TO_TICKS(200));
    ssd1306_clear(); ssd1306_flush();
#endif

#if (CFG_AUDIO_ENABLE == 1)
    if (adpcm_init(NULL, 0, s_video_sem) == 0) {
        /* 初期音量を button の初期ステップから設定 */
        adpcm_set_vol(button_get_vol());
        ESP_LOGI(TAG,"audio OK vol=%u", button_get_vol());
    } else {
        ESP_LOGE(TAG,"audio fail: SW timer");
        start_sw_frame_timer();
    }
#else
    start_sw_frame_timer();
#endif

#if (CFG_OSD_CPU || CFG_OSD_FPS || CFG_OSD_VU)
    xTaskCreate(stats_task, "stats", 2048, NULL, 2, NULL);
#endif
    xTaskCreate(player_task, "player", 6144, NULL, 5, NULL);
}
