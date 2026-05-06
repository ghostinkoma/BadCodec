/**
 * @file  button.c
 * @brief ボタン入力モジュール実装
 * @version v1.0.0
 *
 * ---- 設計 ----
 * button_task が 10ms 周期でポーリング。
 * チャタリング除去: 最後の状態変化から debounce_ms 経過後に確定。
 * 全ボタンは内部プルアップ入力 (押下=LOW)。
 * 未接続ピンは常に HIGH → 非押下扱い → ボタン無しでも正常動作。
 *
 * 音量ボタン:
 *   vol_step を 0-15 でクランプ。押下中 BTN_VOL_SHOW_MS 間だけ
 *   g_btn.vol_show=1 にして main.c 側で表示制御する。
 */
 /* OSD ボタン:
 *   g_btn.osd_mask の bit0(CPU)/bit1(FPS)/bit2(VU) を順に
 *   トグル。1回押すごとに次の組み合わせに移行。
 *   実際の表示/非表示は draw.c の CFG_OSD_xxx ではなく
 *   g_btn.osd_mask で制御するため、config.h の値はデフォルト初期値。
 */

#include "button.h"
#include "driver/gpio.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include <string.h>

static const char *TAG = "BTN";

btn_state_t g_btn;

/* ---- ボタン内部状態 --------------------------------------- */
typedef struct {
    int      gpio;
    uint32_t debounce_ms;
    uint8_t  last_raw;     /* 前回の生の読み値 (1=HIGH=非押下) */
    uint8_t  confirmed;    /* 確定済み状態 (1=HIGH=非押下) */
    uint32_t change_time;  /* 最後に変化した時刻 (ms) */
} btn_ctx_t;

#define NUM_BTNS 4
static btn_ctx_t s_btn[NUM_BTNS];

/* ボタン ID */
#define ID_PLAY   0
#define ID_OSD    1
#define ID_VOLUP  2
#define ID_VOLDN  3

/* ---- 現在時刻 (ms) ---------------------------------------- */
static inline uint32_t now_ms(void)
{
    return (uint32_t)(xTaskGetTickCount() * portTICK_PERIOD_MS);
}

/* ---- GPIO 初期化 ------------------------------------------ */
static void gpio_init_input(int gpio)
{
    gpio_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.pin_bit_mask = (1ULL << gpio);
    cfg.mode         = GPIO_MODE_INPUT;
    cfg.pull_up_en   = GPIO_PULLUP_ENABLE;   /* 内部プルアップ */
    cfg.pull_down_en = GPIO_PULLDOWN_DISABLE;
    cfg.intr_type    = GPIO_INTR_DISABLE;
    gpio_config(&cfg);
}

/* ---- ボタン処理タスク ------------------------------------- */
static void button_task(void *arg)
{
    (void)arg;

    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(10));  /* 10ms ポーリング周期 */
        uint32_t now = now_ms();

        for (int i = 0; i < NUM_BTNS; i++) {
            int raw = gpio_get_level(s_btn[i].gpio);  /* 1=HIGH=非押下 */

            /* 変化検出 */
            if ((uint8_t)raw != s_btn[i].last_raw) {
                s_btn[i].last_raw   = (uint8_t)raw;
                s_btn[i].change_time = now;
            }

            /* チャタリング経過後に確定 */
            if ((now - s_btn[i].change_time) < s_btn[i].debounce_ms) {
                continue;   /* まだ待機中 */
            }

            /* 状態変化なし */
            if ((uint8_t)raw == s_btn[i].confirmed) {
                continue;
            }

            /* ---- 確定: エッジ処理 ---- */
            uint8_t falling = (s_btn[i].confirmed == 1 && raw == 0);
            s_btn[i].confirmed = (uint8_t)raw;

            if (!falling) {
                continue;   /* 立ち上がりエッジは無視 */
            }

            /* ---- 押下確定 ---- */
            switch (i) {
            case ID_PLAY:
                g_btn.paused ^= 1;
                ESP_LOGI(TAG, "PLAY: %s", g_btn.paused ? "PAUSE" : "PLAY");
                break;

            case ID_OSD:
                /* OSD 表示モードを順次切り替え
                 * bit0=CPU  bit1=FPS  bit2=VU
                 * 意味のある 8 パターンを順に切り替える:
                 *   0x07 全表示 (CPU+FPS+VU)
                 *   0x06 FPS+VU
                 *   0x05 CPU+VU
                 *   0x04 VU のみ
                 *   0x03 CPU+FPS
                 *   0x02 FPS のみ
                 *   0x01 CPU のみ
                 *   0x00 非表示
                 * → 0x07 に戻る                                    */
                {
                    static const uint8_t osd_seq[] = {
                        0x07U,  /* CPU+FPS+VU */
                        0x06U,  /* FPS+VU     */
                        0x05U,  /* CPU+VU     */
                        0x04U,  /* VU のみ    */
                        0x03U,  /* CPU+FPS    */
                        0x02U,  /* FPS のみ   */
                        0x01U,  /* CPU のみ   */
                        0x00U,  /* 非表示     */
                    };
                    static const uint8_t osd_seq_len =
                        (uint8_t)(sizeof(osd_seq)/sizeof(osd_seq[0]));
                    /* 現在のマスク値からシーケンス内の位置を探す */
                    uint8_t idx = 0;
                    for (uint8_t k = 0; k < osd_seq_len; k++) {
                        if (osd_seq[k] == g_btn.osd_mask) {
                            idx = (uint8_t)((k + 1U) % osd_seq_len);
                            break;
                        }
                    }
                    g_btn.osd_mask = osd_seq[idx];
                    ESP_LOGI(TAG, "OSD mask: 0x%02X", g_btn.osd_mask);
                }
                break;

            case ID_VOLUP:
                if (g_btn.vol_step < (BTN_VOL_STEPS - 1)) {
                    g_btn.vol_step++;
                }
                g_btn.vol_show     = 1;
                g_btn.vol_show_end = now + BTN_VOL_SHOW_MS;
                ESP_LOGI(TAG, "VOL+: step=%d", g_btn.vol_step);
                break;

            case ID_VOLDN:
                if (g_btn.vol_step > 0) {
                    g_btn.vol_step--;
                }
                g_btn.vol_show     = 1;
                g_btn.vol_show_end = now + BTN_VOL_SHOW_MS;
                ESP_LOGI(TAG, "VOL-: step=%d", g_btn.vol_step);
                break;

            default:
                break;
            }
        }

        /* 音量表示タイムアウト */
        if (g_btn.vol_show && (now >= g_btn.vol_show_end)) {
            g_btn.vol_show = 0;
        }
    }
}

/* ---- 音量ステップ → VOL 値 (0-256) ----------------------- */
/* 16段階の等比ステップ: step 0 = 16, step 15 = 256          */
uint16_t button_get_vol(void)
{
    /* 線形: step * 16 (0→0, 1→16, ..., 15→240, 16→256相当) */
    /* 0を無音にしたい場合は step==0 のとき 0 を返す          */
    if (g_btn.vol_step == 0) {
        return 0U;
    }
    uint16_t v = (uint16_t)g_btn.vol_step * 17U;  /* 1→17 ... 15→255 */
    if (v > 256U) {
        v = 256U;
    }
    return v;
}

/* ---- 初期化 ----------------------------------------------- */
void button_init(uint8_t vol_default_step)
{
    memset(&g_btn, 0, sizeof(g_btn));
    g_btn.osd_mask = 0x07U;   /* 初期値: CPU/FPS/VU 全表示 */
    g_btn.vol_step = vol_default_step;
    g_btn.paused   = 0;

    /* ボタン設定 */
    s_btn[ID_PLAY ].gpio = CFG_BTN_PLAY;
    s_btn[ID_OSD  ].gpio = CFG_BTN_OSD;
    s_btn[ID_VOLUP].gpio = CFG_BTN_VOL_UP;
    s_btn[ID_VOLDN].gpio = CFG_BTN_VOL_DN;

    s_btn[ID_PLAY ].debounce_ms = BTN_DEBOUNCE_PLAY_MS;
    s_btn[ID_OSD  ].debounce_ms = BTN_DEBOUNCE_OSD_MS;
    s_btn[ID_VOLUP].debounce_ms = BTN_DEBOUNCE_VOL_MS;
    s_btn[ID_VOLDN].debounce_ms = BTN_DEBOUNCE_VOL_MS;

    for (int i = 0; i < NUM_BTNS; i++) {
        gpio_init_input(s_btn[i].gpio);
        s_btn[i].last_raw    = 1;   /* 初期=非押下 */
        s_btn[i].confirmed   = 1;
        s_btn[i].change_time = 0;
    }

    xTaskCreate(button_task, "btn", 2048, NULL, 3, NULL);
    ESP_LOGI(TAG, "init: PLAY=GPIO%d OSD=GPIO%d VOL+/-=GPIO%d/%d",
             CFG_BTN_PLAY, CFG_BTN_OSD, CFG_BTN_VOL_UP, CFG_BTN_VOL_DN);
}
