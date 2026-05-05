/**
 * @file  main.c
 * @brief BadCodecPlayer v3.2.0
 *
 * v3.2.0 変更点:
 *   [1] OSD 初期状態を config.h の CFG_OSD_CPU/FPS/VU から生成
 *       (button_init に vol_step と共に osd_mask 初期値を渡す)
 *   [2] ボタン完全統合: PAUSE/VOL/OSD
 *   [3] OSD レイヤー分離: osd_layer_clear/blit で画面乱れなし
 *   [4] PAUSE: 映像+音声同時停止、中央 PAUSE 点滅
 *   [5] CFG_TARGET_FPS_100 対応 (小数点第2位まで)
 *   [6] adpcm_set_frame_us() で μs 精度フレーム同期
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
static uint32_t s_frame_ms = 33U;

/* ---- ソフトタイマ (音声無効時フォールバック) -------------- */
static void frame_timer_cb(TimerHandle_t t)
{
    (void)t;
    xSemaphoreGive(s_video_sem);
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
    memcpy(buf, bad_data + off, len);
    return len;
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
 * stats_task  FPS / CPU 計測
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
 *
 * OSD 描画順序:
 *   1. ssd1306_clear()
 *   2. ssd1306_blit_gram()   映像を g_fb に書く
 *   3. osd_layer_clear()     OSD レイヤーをクリア
 *   4. frame_osd_update()    OSD レイヤーに描画 (g_btn.osd_mask で制御)
 *      osd_draw_vol()        音量表示 (vol_show 中のみ)
 *   5. osd_layer_blit()      OSD を g_fb に INVERT 重畳
 *   6. ssd1306_flush()
 *
 * PAUSE 処理:
 *   adpcm_set_frame_ms(0) で音声 Give を停止
 *   映像は最後のフレームを保持したまま PAUSE 点滅
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

    /* ---- FPS 決定 (小数点第2位対応) ----------------------- */
#if defined(CFG_TARGET_FPS_100) && (CFG_TARGET_FPS_100 != 0)
    uint32_t fps_100 = (uint32_t)CFG_TARGET_FPS_100;
#elif defined(CFG_TARGET_FPS) && (CFG_TARGET_FPS != 0)
    uint32_t fps_100 = (uint32_t)CFG_TARGET_FPS * 100U;
#else
    uint32_t fps_100 = 3000U;   /* デフォルト 30.00fps */
#endif
    if (fps_100 == 0U) { fps_100 = 3000U; }

    /* フレーム間隔 μs: 100000000 / (fps×100) */
    uint32_t frame_us = 100000000U / fps_100;
    if (frame_us == 0U) { frame_us = 1U; }
    s_frame_ms = frame_us / 1000U;
    if (s_frame_ms == 0U) { s_frame_ms = 1U; }

    ESP_LOGI(TAG,"%ux%u %u frames fps=%.2f frame_us=%"PRIu32,
             s_ctx.width, s_ctx.height, s_ctx.total_frames,
             (float)fps_100/100.0f, frame_us);

#if (CFG_AUDIO_ENABLE == 1)
    adpcm_set_frame_us(frame_us);
#endif
    if (s_frame_timer) {
        xTimerChangePeriod(s_frame_timer,
                           pdMS_TO_TICKS(s_frame_ms), portMAX_DELAY);
    }
#else
    ESP_LOGW(TAG,"VIDEO disabled");
#endif /* CFG_VIDEO_ENABLE */

    int was_paused = 0;

    /* ---- メインループ ---- */
    for (;;) {

        /* ---- PAUSE 処理 ----------------------------------- */
        if (g_btn.paused) {
            if (!was_paused) {
#if (CFG_AUDIO_ENABLE == 1)
                adpcm_set_frame_ms(0U);   /* 音声 Give 停止 */
#endif
                /* 積み上がったセマフォを全て捨てる */
                while (xSemaphoreTake(s_video_sem, 0) == pdTRUE) {}
                was_paused = 1;
            }
            vTaskDelay(pdMS_TO_TICKS(100));
#if (CFG_VIDEO_ENABLE == 1)
            /* 最後の映像フレームを保持したまま PAUSE 表示 */
            ssd1306_clear();
            ssd1306_blit_gram(s_ctx.gram,
                              (int)s_ctx.width, (int)s_ctx.height);
            osd_layer_clear();
            osd_draw_pause();   /* 中央点滅 */
            osd_layer_blit();
            ssd1306_flush();
#endif
            continue;
        }

        /* ---- 再生再開 ------------------------------------- */
        if (was_paused) {
            was_paused = 0;
#if (CFG_AUDIO_ENABLE == 1)
            adpcm_set_frame_us(frame_us);   /* 音声 Give 再開 */
#endif
            /* 停止中に積み上がったセマフォを捨てる */
            while (xSemaphoreTake(s_video_sem, 0) == pdTRUE) {}
        }

        /* セマフォ待ち (フレームタイミング) */
        xSemaphoreTake(s_video_sem, portMAX_DELAY);

#if (CFG_VIDEO_ENABLE == 1)
        /* フレームスキップ: 積み上がり分を1フレームだけ消化 */
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
            /* 1. 映像を g_fb に書く */
            ssd1306_clear();
            ssd1306_blit_gram(s_ctx.gram,
                              (int)s_ctx.width, (int)s_ctx.height);

            /* 2. OSD レイヤーをクリアして描画
             *    g_btn.osd_mask: bit0=CPU bit1=FPS bit2=VU
             *    初期値は config.h の CFG_OSD_CPU/FPS/VU から生成済み
             *    0xFFFFFFFF を渡すと frame_osd_update が非表示扱い */
            osd_layer_clear();
            {
                uint32_t show_fps = (g_btn.osd_mask & 0x02U)
                                    ? s_fps_x10 : 0xFFFFFFFFUL;
                uint32_t show_cpu = (g_btn.osd_mask & 0x01U)
                                    ? s_cpu_x10 : 0xFFFFFFFFUL;
                uint16_t show_vu  = (g_btn.osd_mask & 0x04U)
                                    ? adpcm_get_vu() : 0U;
                if (g_btn.osd_mask != 0U) {
                    frame_osd_update(frame, show_fps, show_cpu, show_vu);
                }
                /* 音量バー (1.5秒間表示) */
                if (g_btn.vol_show) {
                    osd_draw_vol(g_btn.vol_step);
                }
            }

            /* 3. OSD を映像に INVERT 重畳 */
            osd_layer_blit();

            /* 4. OLED 転送 */
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
    ESP_LOGI(TAG,"BadCodecPlayer v3.2.0  VIDEO=%d AUDIO=%d",
             CFG_VIDEO_ENABLE, CFG_AUDIO_ENABLE);

    s_video_sem = xSemaphoreCreateBinary();
    if (!s_video_sem) { esp_restart(); }

    /* ---- ボタン初期化 -------------------------------------
     * OSD 初期マスクは config.h の CFG_OSD_CPU/FPS/VU から生成。
     * button_init() 内で以下を実行:
     *   g_btn.osd_mask = (CFG_OSD_CPU?0x01:0)
     *                  | (CFG_OSD_FPS?0x02:0)
     *                  | (CFG_OSD_VU ?0x04:0)
     * ------------------------------------------------------- */
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
        ESP_LOGI(TAG,"audio OK");
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
