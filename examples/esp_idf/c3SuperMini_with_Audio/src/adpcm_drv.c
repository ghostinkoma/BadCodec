/**
 * @file  adpcm_drv.c
 * @brief IMA ADPCM decoder + Software Sigma-Delta PDM audio output
 * @version v1.1.0
 *
 * --- 音質設計 ---
 *
 * v0.9.x の問題:
 *   ISR @ 16kHz で GPIO に PCM符号をそのまま出力。
 *   等価帯域幅 8kHz しかなく低音が全く再現されない。
 *
 * v1.0.x の問題:
 *   ESP-IDF ハードウェア SDM ドライバ (driver/sdm.h) を使用したが
 *   sdm_channel_set_duty() は ISR から呼ぶ設計ではなく音が出なかった。
 *
 * v1.1.0 (本実装):
 *   ソフトウェア 1次 ΣΔ変調 + gptimer @ 16kHz (ISR内でOSR=32ループ)
 *
 *   動作原理:
 *     ISR が 16kHz で発火するたびに SDM_OSR=32 回のΣΔループを実行。
 *     実効 PDM クロック = 16kHz × 32 = 512kHz。
 *     ISR 発火は 16kHz なので Interrupt Watchdog に問題なし。
 *
 *       accumulator += current_pcm_sample
 *       if (accumulator >= 0):
 *           GPIO_P = HIGH, GPIO_N = LOW
 *           accumulator -= 32767
 *       else:
 *           GPIO_P = LOW,  GPIO_N = HIGH
 *           accumulator += 32767
 *     このループを SDM_OSR(32) 回実行して1サンプル分の PDM を生成。
 *
 *   実効 OSR = 32x → 旧版 (符号出力 OSR=1) より大幅な音質改善
 *   低音 (20-200Hz) が正確に再現される
 *
 *   GPIO 制御: REG_WRITE (soc/gpio_reg.h) で IRAM 上から直接レジスタアクセス。
 *   sdm.h / sdm_channel_set_duty() は ISR 非対応のため使用しない。
 *
 * --- 接続 ---
 *   CFG_AUDIO_PIN_P (GPIO10) → R=1kΩ → ┬─ C=10nF → GND  (LPF fc≈16kHz)
 *                                        └─ C=10μF → アンプ入力 (カップリング)
 *   CFG_AUDIO_PIN_N (GPIO3)  → 差動出力として使用
 *   (シングルエンドの場合は GPIO10 のみ使用し GPIO3 は未接続でも可)
 */

/* ---- adpcm4.h はこのTUのみでインクルード ---- */
#include "adpcm4.h"
#include "adpcm_drv.h"

#include "driver/gpio.h"
#include "driver/gptimer.h"
#include "esp_log.h"
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "soc/gpio_reg.h"   /* GPIO_OUT_W1TS_REG / GPIO_OUT_W1TC_REG */

#include <string.h>
#include <inttypes.h>

static const char * const TAG = "ADPCM";

/* ============================================================
 * ISR 設計 (v1.2.0):
 *   gptimer を 160kHz (= SR × SDM_OSR) で発火。
 *   ISR 毎に 1bit だけ ΣΔ演算して GPIO を1回切り替える。
 *   OSRカウンタが SDM_OSR(10) に達したら次のPCMサンプルを取得。
 *
 *   80MHz / 160kHz = 500 (整数分周、誤差ゼロ)
 *   ISR 予算: 160MHz / 160kHz = 1000 cycles/ISR → 余裕十分
 *   OSR = 10x
 *
 * 【旧 v1.1.0 の問題と修正】
 *   ISR 16kHz + ループ32回 → 32回の切り替えが ~3μs で完了し
 *   残り 59.5μs は最後の GPIO 状態を保持 → LPF が DC だけ見る
 *   → ΣΔ効果なし、ホワイトノイズ発生。
 *   正しくは ISR ごとに1bit 出力し、切り替え間隔を均等にする。
 * ============================================================ */
#define SDM_OSR   10U   /* OSR: 160kHz / 16kHz = 10 */

/* ============================================================
 * IMA ADPCM テーブル
 * ============================================================ */
static const int16_t s_step_table[89] = {
       7,    8,    9,   10,   11,   12,   13,   14,
      16,   17,   19,   21,   23,   25,   28,   31,
      34,   37,   41,   45,   50,   55,   60,   66,
      73,   80,   88,   97,  107,  118,  130,  143,
     157,  173,  190,  209,  230,  253,  279,  307,
     337,  371,  408,  449,  494,  544,  598,  658,
     724,  796,  876,  963, 1060, 1163, 1282, 1411,
    1552, 1707, 1878, 2066, 2272, 2499, 2749, 3024,
    3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484,
    7132, 7845, 8630, 9493,10442,11487,12635,13899,
   15289,16818,18500,20350,22385,24623,27086,29794,
   32767
};

static const int8_t s_idx_table[8] = {
    -1, -1, -1, -1, 2, 4, 6, 8
};

/* ============================================================
 * リングバッファ (decodeタスク → ISR)
 * ============================================================ */
static volatile int16_t  s_ring[ADPCM_RING_SIZE];
static volatile uint32_t s_ring_wr;
static volatile uint32_t s_ring_rd;

static inline uint32_t ring_used(void)
{
    return (s_ring_wr - s_ring_rd) & (ADPCM_RING_SIZE - 1U);
}
static inline int ring_full(void)
{
    return ring_used() >= (ADPCM_RING_SIZE - 1U);
}
static inline int ring_empty(void)
{
    return s_ring_wr == s_ring_rd;
}

/* ============================================================
 * 静的状態
 * ============================================================ */
static const uint8_t    *s_wav_data;
static uint32_t          s_wav_size;
static uint32_t          s_data_offset;
static uint32_t          s_data_size;
static uint32_t          s_block_align;
static uint32_t          s_sample_rate;
static uint32_t          s_read_pos;

static SemaphoreHandle_t s_video_sem;
static gptimer_handle_t  s_gptimer;

/* ISR 同期カウンタ */
static volatile uint32_t s_isr_osr_cnt;   /* OSRカウンタ (0〜SDM_OSR-1) */
static volatile int16_t  s_cur_sample;    /* 現在のPCMサンプル */
/* 2次ΣΔ変調アキュムレータ
 *   e1: 1段目積分器 (入力 - フィードバック)
 *   e2: 2段目積分器 (e1   - フィードバック)
 *   量子化ノイズを高周波に押し込み可聴域の SNR を大幅改善する */
static volatile int32_t  s_sd_e1;         /* 1段目アキュムレータ */
static volatile int32_t  s_sd_e2;         /* 2段目アキュムレータ */
static volatile uint32_t s_isr_samples;   /* 消費サンプル数 */
static volatile uint32_t s_frames_given;

/* GPIO ビットマスク (ISR 高速アクセス用、IRAM 上) */
static uint32_t s_mask_p;
static uint32_t s_mask_n;

/* ============================================================
 * WAV ヘッダパーサ
 * ============================================================ */
static int wav_parse(void)
{
    const uint8_t *d = s_wav_data;

    if (s_wav_size < 44U) {
        ESP_LOGE(TAG, "WAV too small (%"PRIu32")", s_wav_size);
        return -1;
    }
    if (d[0]!='R'||d[1]!='I'||d[2]!='F'||d[3]!='F') {
        ESP_LOGE(TAG, "Missing RIFF magic"); return -1;
    }
    if (d[8]!='W'||d[9]!='A'||d[10]!='V'||d[11]!='E') {
        ESP_LOGE(TAG, "Missing WAVE magic"); return -1;
    }

    uint32_t pos = 12U;
    uint32_t af = 0U, sr = 0U, ba = 0U;
    int got_fmt = 0, got_data = 0;

    while ((pos + 8U) <= s_wav_size) {
        uint32_t csz = (uint32_t)d[pos+4]
                     | ((uint32_t)d[pos+5] <<  8)
                     | ((uint32_t)d[pos+6] << 16)
                     | ((uint32_t)d[pos+7] << 24);

        if (d[pos]=='f'&&d[pos+1]=='m'&&d[pos+2]=='t'&&d[pos+3]==' ') {
            af = (uint32_t)d[pos+ 8] | ((uint32_t)d[pos+ 9]<<8);
            sr = (uint32_t)d[pos+12] | ((uint32_t)d[pos+13]<<8)
               | ((uint32_t)d[pos+14]<<16) | ((uint32_t)d[pos+15]<<24);
            ba = (uint32_t)d[pos+20] | ((uint32_t)d[pos+21]<<8);
            got_fmt = 1;
            ESP_LOGI(TAG, "fmt: format=0x%04"PRIu32" SR=%"PRIu32" BA=%"PRIu32,
                     af, sr, ba);
        } else if (d[pos]=='d'&&d[pos+1]=='a'&&d[pos+2]=='t'&&d[pos+3]=='a') {
            s_data_offset = pos + 8U;
            s_data_size   = csz;
            got_data = 1;
            ESP_LOGI(TAG, "data: offset=%"PRIu32" size=%"PRIu32,
                     s_data_offset, s_data_size);
        }
        pos += 8U + csz;
        if (csz & 1U) { pos++; }
        if (got_fmt && got_data) { break; }
    }

    if (!got_fmt || !got_data) {
        ESP_LOGE(TAG, "fmt/data chunk not found"); return -1;
    }
    if (af != (uint32_t)WAV_FMT_IMA_ADPCM) {
        ESP_LOGE(TAG, "Not IMA ADPCM (0x%04"PRIu32")", af); return -1;
    }
    s_block_align = ba;
    s_sample_rate = sr;
    s_read_pos    = s_data_offset;
    ESP_LOGI(TAG, "WAV OK  sr=%"PRIu32" ba=%"PRIu32" sz=%"PRIu32,
             sr, ba, s_data_size);
    return 0;
}

/* ============================================================
 * IMA ADPCM ニブルデコーダ
 * ============================================================ */
static IRAM_ATTR int16_t adpcm_nibble(adpcm_state_t *st, uint8_t n)
{
    int16_t step = s_step_table[st->step_index];
    int32_t diff = (int32_t)step >> 3;
    if (n & 4U) { diff += (int32_t)step; }
    if (n & 2U) { diff += (int32_t)step >> 1; }
    if (n & 1U) { diff += (int32_t)step >> 2; }
    if (n & 8U) { diff = -diff; }
    int32_t pred = (int32_t)st->predictor + diff;
    if      (pred >  32767) { pred =  32767; }
    else if (pred < -32768) { pred = -32768; }
    st->predictor = (int16_t)pred;
    int8_t idx = st->step_index + s_idx_table[n & 7U];
    if      (idx <  0) { idx =  0; }
    else if (idx > 88) { idx = 88; }
    st->step_index = idx;
    return st->predictor;
}

/* ============================================================
 * リングバッファへのデコード
 * ============================================================ */
static void decode_block_to_ring(const uint8_t *blk, uint32_t bytes)
{
    if (bytes < 4U) { return; }

    adpcm_state_t st;
    st.predictor  = (int16_t)((uint16_t)blk[0] | ((uint16_t)blk[1]<<8));
    st.step_index = (int8_t)blk[2];
    if (st.step_index <  0) { st.step_index =  0; }
    if (st.step_index > 88) { st.step_index = 88; }

#define RING_PUSH(raw_sample) do {                                      \
    int16_t _s = (int16_t)(((int32_t)(raw_sample)                       \
                             * (int32_t)CFG_AUDIO_VOL) >> 8);           \
    while (ring_full()) { vTaskDelay(1); }                              \
    s_ring[s_ring_wr & (ADPCM_RING_SIZE - 1U)] = _s;                    \
    s_ring_wr++;                                                         \
} while (0)

    RING_PUSH(st.predictor);
    for (uint32_t i = 4U; i < bytes; i++) {
        int16_t s0 = adpcm_nibble(&st,  blk[i]       & 0x0FU);
        int16_t s1 = adpcm_nibble(&st, (blk[i] >> 4) & 0x0FU);
        RING_PUSH(s0);
        RING_PUSH(s1);
    }
#undef RING_PUSH
}

/* ============================================================
 * gptimer ISR — 160kHz (= s_sample_rate × SDM_OSR) で発火
 *
 * 2次ソフトウェアΣΔ変調 (ノイズシェーピング強化版)
 *
 * 1次ΣΔ (OSR=10): SNR ≈ 21 dB → 不十分
 * 2次ΣΔ (OSR=10): SNR ≈ 43 dB → 実用的な音楽品質
 *
 * アルゴリズム:
 *   fb  = out_prev ? +32767 : -32767  (1bit フィードバック)
 *   e1 += sample - fb                 (1段目積分)
 *   e2 += e1     - fb                 (2段目積分)
 *   out = (e2 >= 0)                   (量子化)
 *   GPIO_P = out
 *
 * OSRカウンタが SDM_OSR(10) に達したら ring から次サンプルを取得。
 * ============================================================ */
static IRAM_ATTR bool sd_isr_cb(gptimer_handle_t timer,
                                 const gptimer_alarm_event_data_t *edata,
                                 void *user_ctx)
{
    (void)timer; (void)edata; (void)user_ctx;
    BaseType_t woken = pdFALSE;

    /* OSR カウンタ: SDM_OSR(10) ごとに次サンプルを取得 */
    s_isr_osr_cnt++;
    if (s_isr_osr_cnt >= SDM_OSR) {
        s_isr_osr_cnt = 0U;
        if (!ring_empty()) {
            s_cur_sample = s_ring[s_ring_rd & (ADPCM_RING_SIZE - 1U)];
            s_ring_rd++;
        } else {
            s_cur_sample = 0;   /* アンダーラン → 無音 */
        }
        /* フレーム同期 */
        s_isr_samples++;
        uint32_t spf = (s_sample_rate * (uint32_t)CFG_FRAME_MS) / 1000U;
        if (spf > 0U) {
            uint32_t frames_now = s_isr_samples / spf;
            while (s_frames_given < frames_now) {
                xSemaphoreGiveFromISR(s_video_sem, &woken);
                s_frames_given++;
            }
        }
    }

    /* 2次ΣΔ変調 → GPIO を1bit 出力
     *
     * fb = 現在の出力に対するフィードバック値
     *      出力 HIGH (+1) → fb = +32767
     *      出力 LOW  (-1) → fb = -32767
     *
     * e1 += sample - fb   (1段目: 入力 - フィードバック を積分)
     * e2 += e1 - fb        (2段目: e1  - フィードバック を積分)
     * out = (e2 >= 0)      (量子化)
     *
     * 2段階の積分により量子化ノイズのスペクトルが高周波側に
     * より急峻にシフトされ、可聴域 (0〜16kHz/2=8kHz) の SNR が向上する。
     */
    int32_t sample = (int32_t)s_cur_sample;
    int32_t e1     = s_sd_e1;
    int32_t e2     = s_sd_e2;

    /* 前回の出力に基づくフィードバック */
    int32_t fb = (e2 >= 0) ? 32767 : -32767;

    e1 += sample - fb;
    e2 += e1     - fb;

    /* オーバーフロー抑制: 積分器が発散しないようにクランプ */
    if      (e1 >  0x400000) { e1 =  0x400000; }
    else if (e1 < -0x400000) { e1 = -0x400000; }
    if      (e2 >  0x400000) { e2 =  0x400000; }
    else if (e2 < -0x400000) { e2 = -0x400000; }

    s_sd_e1 = e1;
    s_sd_e2 = e2;

    /* GPIO 出力 */
#if (CFG_AUDIO_DIFFERENTIAL == 1)
    if (e2 >= 0) {
        REG_WRITE(GPIO_OUT_W1TS_REG, s_mask_p);
        REG_WRITE(GPIO_OUT_W1TC_REG, s_mask_n);
    } else {
        REG_WRITE(GPIO_OUT_W1TC_REG, s_mask_p);
        REG_WRITE(GPIO_OUT_W1TS_REG, s_mask_n);
    }
#else
    if (e2 >= 0) {
        REG_WRITE(GPIO_OUT_W1TS_REG, s_mask_p);
    } else {
        REG_WRITE(GPIO_OUT_W1TC_REG, s_mask_p);
    }
#endif

    return (woken == pdTRUE);
}

/* ============================================================
 * オーディオデコードタスク (priority=4 < player_task=5)
 * ============================================================ */
static void adpcm_task(void *arg)
{
    (void)arg;
    ESP_LOGI(TAG, "audio task started  core=%d  pri=%d",
             xPortGetCoreID(), uxTaskPriorityGet(NULL));

    for (;;) {
        if (s_read_pos >= (s_data_offset + s_data_size)) {
            vTaskDelay(pdMS_TO_TICKS(50U));
            continue;
        }
        uint32_t remaining = (s_data_offset + s_data_size) - s_read_pos;
        uint32_t blk_bytes = (s_block_align > 0U) ? s_block_align : 512U;
        if (blk_bytes > remaining) { blk_bytes = remaining; }

        decode_block_to_ring(s_wav_data + s_read_pos, blk_bytes);
        s_read_pos += blk_bytes;
    }
}

/* ============================================================
 * adpcm_init — public API
 * ============================================================ */
int adpcm_init(const uint8_t    *adpcm_data,
               uint32_t          adpcm_size,
               SemaphoreHandle_t video_sem)
{
    if (adpcm_data == NULL) {
        adpcm_data = bad_audio_data;
        adpcm_size = BAD_AUDIO_SIZE;
    }

    s_wav_data     = adpcm_data;
    s_wav_size     = adpcm_size;
    s_video_sem    = video_sem;
    s_ring_wr      = 0U;
    s_ring_rd      = 0U;
    s_isr_osr_cnt  = 0U;
    s_cur_sample   = 0;
    s_sd_e1        = 0;
    s_sd_e2        = 0;
    s_isr_samples  = 0U;
    s_frames_given = 0U;

    if (wav_parse() != 0) { return -1; }

    uint32_t isr_hz = s_sample_rate * (uint32_t)SDM_OSR;  /* 16000×10=160000 Hz */
    ESP_LOGI(TAG, "SR=%"PRIu32"  OSR=%u  ISR=%"PRIu32"Hz  SPF=%"PRIu32,
             s_sample_rate, SDM_OSR, isr_hz,
             (s_sample_rate * (uint32_t)CFG_FRAME_MS) / 1000U);

    /* GPIO 設定 */
    s_mask_p = (1U << (uint32_t)CFG_AUDIO_PIN_P);
    s_mask_n = (1U << (uint32_t)CFG_AUDIO_PIN_N);

    gpio_config_t gc;
    memset(&gc, 0, sizeof(gc));
#if (CFG_AUDIO_DIFFERENTIAL == 1)
    /* 差動: PIN_P + PIN_N 両方を OUTPUT に設定 */
    gc.pin_bit_mask = ((uint64_t)1U << CFG_AUDIO_PIN_P)
                    | ((uint64_t)1U << CFG_AUDIO_PIN_N);
    gc.mode         = GPIO_MODE_OUTPUT;
    gc.pull_up_en   = GPIO_PULLUP_DISABLE;
    gc.pull_down_en = GPIO_PULLDOWN_DISABLE;
    gc.intr_type    = GPIO_INTR_DISABLE;
    ESP_ERROR_CHECK(gpio_config(&gc));
    ESP_ERROR_CHECK(gpio_set_level(CFG_AUDIO_PIN_P, 0U));
    ESP_ERROR_CHECK(gpio_set_level(CFG_AUDIO_PIN_N, 1U));
    ESP_LOGI(TAG, "GPIO mode: DIFFERENTIAL  P=GPIO%d  N=GPIO%d",
             CFG_AUDIO_PIN_P, CFG_AUDIO_PIN_N);
#else
    /* シングルエンド: PIN_P のみ OUTPUT。PIN_N は固定 LOW (使用しない) */
    gc.pin_bit_mask = ((uint64_t)1U << CFG_AUDIO_PIN_P);
    gc.mode         = GPIO_MODE_OUTPUT;
    gc.pull_up_en   = GPIO_PULLUP_DISABLE;
    gc.pull_down_en = GPIO_PULLDOWN_DISABLE;
    gc.intr_type    = GPIO_INTR_DISABLE;
    ESP_ERROR_CHECK(gpio_config(&gc));
    ESP_ERROR_CHECK(gpio_set_level(CFG_AUDIO_PIN_P, 0U));
    ESP_LOGI(TAG, "GPIO mode: SINGLE-ENDED  P=GPIO%d  (N=GPIO%d unused)",
             CFG_AUDIO_PIN_P, CFG_AUDIO_PIN_N);
#endif

    /* gptimer: SR * OSR Hz で発火 */
    gptimer_config_t tcfg;
    memset(&tcfg, 0, sizeof(tcfg));
    tcfg.clk_src       = GPTIMER_CLK_SRC_DEFAULT;
    tcfg.direction     = GPTIMER_COUNT_UP;
    tcfg.resolution_hz = isr_hz;

    esp_err_t err = gptimer_new_timer(&tcfg, &s_gptimer);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "gptimer_new_timer failed: 0x%x", err);
        return -1;
    }

    gptimer_alarm_config_t acfg;
    memset(&acfg, 0, sizeof(acfg));
    acfg.alarm_count                = 1U;
    acfg.reload_count               = 0U;
    acfg.flags.auto_reload_on_alarm = 1U;
    ESP_ERROR_CHECK(gptimer_set_alarm_action(s_gptimer, &acfg));

    gptimer_event_callbacks_t cbs;
    memset(&cbs, 0, sizeof(cbs));
    cbs.on_alarm = sd_isr_cb;
    ESP_ERROR_CHECK(gptimer_register_event_callbacks(s_gptimer, &cbs, NULL));
    ESP_ERROR_CHECK(gptimer_enable(s_gptimer));

    /* ---- ring バッファの事前充填 --------------------------------
     * gptimer (ISR) を開始する前に ring を満杯にしておく。
     * これにより起動直後の ring underrun によるグリッチを防ぐ。
     *
     * RING_SIZE=131072 samples = 8.2秒分を事前充填する。
     * adpcm_init() は app_main タスクコンテキスト (FreeRTOS スケジューラ動作中)
     * で呼ばれるため vTaskDelay を使わず直接デコードループを回す。
     * (スケジューラ開始前の場合は vTaskDelay が使えないが、
     *  ここでは app_main 内 = スケジューラ起動後なので問題なし)
     * ------------------------------------------------------------ */
    ESP_LOGI(TAG, "pre-filling ring buffer (%u samples)...", ADPCM_RING_SIZE);
    {
        uint32_t filled = 0U;
        while (filled < (ADPCM_RING_SIZE - 1U)) {
            if (s_read_pos >= (s_data_offset + s_data_size)) {
                break;  /* WAV データ終端 */
            }
            uint32_t remaining = (s_data_offset + s_data_size) - s_read_pos;
            uint32_t blk = (s_block_align > 0U) ? s_block_align : 512U;
            if (blk > remaining) { blk = remaining; }

            /* ring が満杯になるまでデコード */
            const uint8_t *p = s_wav_data + s_read_pos;

            adpcm_state_t st;
            st.predictor  = (int16_t)((uint16_t)p[0] | ((uint16_t)p[1]<<8));
            st.step_index = (int8_t)p[2];
            if (st.step_index <  0) { st.step_index =  0; }
            if (st.step_index > 88) { st.step_index = 88; }

            /* ヘッダサンプル */
            if (!ring_full()) {
                int16_t s = (int16_t)(((int32_t)st.predictor
                                       * (int32_t)CFG_AUDIO_VOL) >> 8);
                s_ring[s_ring_wr & (ADPCM_RING_SIZE - 1U)] = s;
                s_ring_wr++;
                filled++;
            }
            /* ニブルサンプル */
            for (uint32_t i = 4U; i < blk && !ring_full(); i++) {
                int16_t s0 = adpcm_nibble(&st, p[i] & 0x0FU);
                int16_t s1 = adpcm_nibble(&st, (p[i] >> 4) & 0x0FU);
                int16_t v0 = (int16_t)(((int32_t)s0 * (int32_t)CFG_AUDIO_VOL) >> 8);
                int16_t v1 = (int16_t)(((int32_t)s1 * (int32_t)CFG_AUDIO_VOL) >> 8);
                s_ring[s_ring_wr & (ADPCM_RING_SIZE - 1U)] = v0; s_ring_wr++; filled++;
                if (ring_full()) break;
                s_ring[s_ring_wr & (ADPCM_RING_SIZE - 1U)] = v1; s_ring_wr++; filled++;
            }
            s_read_pos += blk;
        }
        ESP_LOGI(TAG, "pre-fill done: %"PRIu32" samples (%.1f sec)",
                 filled, (float)filled / (float)s_sample_rate);
    }

    /* gptimer 開始: ring が充填済みなので起動直後からノイズなし */
    ESP_ERROR_CHECK(gptimer_start(s_gptimer));

    xTaskCreatePinnedToCore(adpcm_task, "adpcm",
                            4096U, NULL,
                            4,    /* priority=4 < player_task(5)
                                   * adpcm_task がプリエンプトして
                                   * player_task の I2C 転送を中断しないようにする。
                                   * ring_full 時は vTaskDelay(1) で必ず
                                   * player_task に制御を渡す。
                                   * 消費レート 16kHz × 35ms = 560 samples。
                                   * RING_SIZE=4096 で 221ms 分の余裕がある。 */
                            NULL,
                            0);   /* ESP32-C3 はシングルコア */

    ESP_LOGI(TAG, "adpcm_init OK  ISR@%"PRIu32"Hz", isr_hz);
    return 0;
}

/* ============================================================
 * adpcm_rewind — public API
 * ============================================================ */
void adpcm_rewind(void)
{
    gptimer_stop(s_gptimer);
    gptimer_disable(s_gptimer);

    s_read_pos     = s_data_offset;
    s_ring_wr      = 0U;
    s_ring_rd      = 0U;
    s_isr_osr_cnt  = 0U;
    s_cur_sample   = 0;
    s_sd_e1        = 0;
    s_sd_e2        = 0;
    s_isr_samples  = 0U;
    s_frames_given = 0U;

    /* GPIO をアイドル状態に戻す */
#if (CFG_AUDIO_DIFFERENTIAL == 1)
    REG_WRITE(GPIO_OUT_W1TC_REG, s_mask_p);
    REG_WRITE(GPIO_OUT_W1TS_REG, s_mask_n);
#else
    REG_WRITE(GPIO_OUT_W1TC_REG, s_mask_p);
#endif

    gptimer_enable(s_gptimer);
    gptimer_start(s_gptimer);
    ESP_LOGI(TAG, "rewind");
}
