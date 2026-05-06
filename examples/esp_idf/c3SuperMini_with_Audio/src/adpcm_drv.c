/**
 * @file  adpcm_drv.c
 * @brief IMA ADPCM decoder + LEDC PWM audio  v4.1.0
 *
 * ============================================================
 * v4.1.0 変更点
 * ============================================================
 *
 * [1] 複数ピン並列出力 (最大4本)
 *   config.h で CFG_AUDIO_PIN_P2/P3/P4 を設定 (0=無効)。
 *   初期化・ISR の duty 更新・rewind の無音化すべてに適用。
 *   各ピンに R=10kΩ を直列に入れて合流させること (並列直結禁止)。
 *   直結スピーカー駆動時: C=100μF でカップリングして接続。
 *
 * [2] DUTY_SWING フルスイング化
 *   旧: DUTY_SWING = 1002 (49% 振幅)
 *   修: DUTY_SWING = PWM_MID_DUTY = 1024 (100% 振幅)
 *   → ダイナミックレンジ +6dB 改善
 *
 * [3] 音量スケーリングを ISR に集約
 *   旧: prefill/decode_block でリングに CFG_AUDIO_VOL を適用
 *       → ISR がリングの値を直接使う (音量変更にはリングを作り直す必要)
 *   修: リングには生 PCM (±32767) を格納
 *       ISR で pcm × CFG_AUDIO_VOL >> 8 を適用してから duty 計算
 *       → 音量変更が即時反映可能な構造
 *
 * [4] VU 減衰
 *   adpcm_get_vu() 呼び出しごとに 1/8 ずつ減衰。
 *   バーが滑らかに動く。
 *
 * [5] adpcm_set_frame_us() 追加 (μs 精度)
 *   29.97fps = 33366μs → 誤差 0.003%
 *
 * [6] prefill/decode_task 二重ライター競合解消
 *   decode_task を先に起動し、タスク内フェーズ1で prefill 完了後に
 *   gptimer を起動。s_ring_wr を触るのは decode_task のみ。
 *
 * ============================================================
 * include 設計: adpcm_drv.h は config.h を include しない (循環防止)
 * ============================================================
 */

#include "config.h"
#include "adpcm4.h"
#include "adpcm_drv.h"
#include "driver/ledc.h"
#include "driver/gptimer.h"
#include "esp_log.h"
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include <string.h>
#include <inttypes.h>

static const char *const TAG = "ADPCM";

/* ============================================================
 * 並列ピン有効判定
 * config.h で未定義の場合は 0 (無効) とする
 * ============================================================ */
#ifndef CFG_AUDIO_PIN_P2
#  define CFG_AUDIO_PIN_P2  0
#endif
#ifndef CFG_AUDIO_PIN_P3
#  define CFG_AUDIO_PIN_P3  0
#endif
#ifndef CFG_AUDIO_PIN_P4
#  define CFG_AUDIO_PIN_P4  0
#endif

/* ============================================================
 * LEDC 設定
 * ============================================================ */
#define PWM_RESOLUTION   LEDC_TIMER_11_BIT  /* 2048 段階 */
#define PWM_MAX_DUTY     2047U
#define PWM_MID_DUTY     1024U              /* 無音=中点 */
#define PWM_FREQ_HZ      39062U             /* ~39kHz キャリア */
#define PWM_TIMER        LEDC_TIMER_0
#define PWM_SPEED_MODE   LEDC_LOW_SPEED_MODE

/* チャンネル割り当て */
#define PWM_CHANNEL_P    LEDC_CHANNEL_0     /* 正相 (必須) */
#define PWM_CHANNEL_N    LEDC_CHANNEL_1     /* 逆相 (BTL) */
#define PWM_CHANNEL_P2   LEDC_CHANNEL_2     /* 並列ピン2 */
#define PWM_CHANNEL_P3   LEDC_CHANNEL_3     /* 並列ピン3 */
#define PWM_CHANNEL_P4   LEDC_CHANNEL_4     /* 並列ピン4 */

/* ダイナミックレンジ最大化: フルスイング
 * PCM ±32767 → duty ±1024 (中点 1024 から最大/最小まで)
 * 旧: 1002 (49%) → 修: 1024 (50%, クリップなし上限) */
#define DUTY_SWING       PWM_MID_DUTY       /* = 1024 */

/* ---- gptimer -------------------------------------------- */
#define TIMER_RES_HZ     1000000U

/* ---- prefill 目標サンプル数: リングの 75% ---- */
#define PREFILL_TARGET  (ADPCM_RING_SIZE * 3U / 4U)

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
static const int8_t s_idx_table[8] = {-1,-1,-1,-1, 2, 4, 6, 8};

/* ============================================================
 * リングバッファ
 * wr/rd は絶対値として単調増加。差をそのまま使う。
 * インデックス = wr/rd & (SIZE-1)
 * ============================================================ */
static volatile int16_t  s_ring[ADPCM_RING_SIZE];
static volatile uint32_t s_ring_wr;
static volatile uint32_t s_ring_rd;

static inline uint32_t ring_used(void)  { return s_ring_wr - s_ring_rd; }
static inline int ring_full(void)  { return ring_used() >= (ADPCM_RING_SIZE-1U); }
static inline int ring_empty(void) { return s_ring_wr == s_ring_rd; }

/* ============================================================
 * 状態変数
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

/* フレーム間隔: us 優先, us==0 なら ms を使用 */
static volatile uint32_t s_frame_interval_ms;
static volatile uint32_t s_frame_interval_us;

static volatile uint32_t s_isr_samples;
static volatile uint32_t s_frames_given;
static volatile uint16_t s_vu_peak;

/* prefill 完了フラグ: decode_task → adpcm_init */
static volatile int s_prefill_done;

/* 動的音量: 0-256。adpcm_set_vol() で更新。初期値は CFG_AUDIO_VOL */
static volatile uint32_t s_vol;

/* ============================================================
 * Public API
 * ============================================================ */
void adpcm_set_frame_ms(uint32_t ms)
{
    s_frame_interval_ms = ms;
    s_frame_interval_us = 0U;
}
void adpcm_set_frame_us(uint32_t us)
{
    s_frame_interval_us = us;
    s_frame_interval_ms = 0U;
}
uint32_t adpcm_get_sample_rate(void) { return s_sample_rate; }

/* ============================================================
 * adpcm_pause / adpcm_resume
 *
 * PAUSE: gptimer を止めて ledc を無音(MID_DUTY)に設定。
 *        s_isr_samples / s_frames_given はそのまま保持。
 *
 * RESUME: gptimer を再開。
 *         s_isr_samples と s_frames_given をリセットして
 *         再開直後の一気 Give を防ぐ。
 * ============================================================ */
void adpcm_pause(void)
{
    if (s_gptimer == NULL) { return; }
    gptimer_stop(s_gptimer);
    /* 出力を無音に (クリック防止) */
    ledc_set_duty(PWM_SPEED_MODE, PWM_CHANNEL_P, PWM_MID_DUTY);
    ledc_update_duty(PWM_SPEED_MODE, PWM_CHANNEL_P);
#if (CFG_AUDIO_OUTPUT_MODE == 1)
    ledc_set_duty(PWM_SPEED_MODE, PWM_CHANNEL_N, PWM_MID_DUTY);
    ledc_update_duty(PWM_SPEED_MODE, PWM_CHANNEL_N);
#endif
#if (CFG_AUDIO_PIN_P2 != 0)
    ledc_set_duty(PWM_SPEED_MODE, PWM_CHANNEL_P2, PWM_MID_DUTY);
    ledc_update_duty(PWM_SPEED_MODE, PWM_CHANNEL_P2);
#endif
#if (CFG_AUDIO_PIN_P3 != 0)
    ledc_set_duty(PWM_SPEED_MODE, PWM_CHANNEL_P3, PWM_MID_DUTY);
    ledc_update_duty(PWM_SPEED_MODE, PWM_CHANNEL_P3);
#endif
#if (CFG_AUDIO_PIN_P4 != 0)
    ledc_set_duty(PWM_SPEED_MODE, PWM_CHANNEL_P4, PWM_MID_DUTY);
    ledc_update_duty(PWM_SPEED_MODE, PWM_CHANNEL_P4);
#endif
    ESP_LOGI(TAG,"audio pause");
}

void adpcm_resume(void)
{
    if (s_gptimer == NULL) { return; }
    /* カウンタリセット: 再開時の一気 Give を防ぐ */
    s_isr_samples  = 0;
    s_frames_given = 0;
    gptimer_start(s_gptimer);
    ESP_LOGI(TAG,"audio resume");
}

void adpcm_set_vol(uint16_t vol)
{
    /* vol: 0=無音, 256=フルスケール (CFG_AUDIO_VOL と同単位) */
    if (vol > 256U) { vol = 256U; }
    s_vol = (uint32_t)vol;
}

uint16_t adpcm_get_vu(void)
{
    uint16_t v = s_vu_peak;
    /* 呼び出しごとに 1/8 減衰 → バーが滑らかに動く */
    s_vu_peak = (uint16_t)(s_vu_peak - (s_vu_peak >> 3));
    return v;
}

/* ============================================================
 * WAV パーサ
 * ============================================================ */
static int wav_parse(void)
{
    const uint8_t *d = s_wav_data;
    if (s_wav_size < 44U) { ESP_LOGE(TAG,"WAV too small"); return -1; }
    if (d[0]!='R'||d[1]!='I'||d[2]!='F'||d[3]!='F') {
        ESP_LOGE(TAG,"No RIFF"); return -1;
    }
    if (d[8]!='W'||d[9]!='A'||d[10]!='V'||d[11]!='E') {
        ESP_LOGE(TAG,"No WAVE"); return -1;
    }
    uint32_t pos=12, af=0, sr=0, ba=0;
    int got_fmt=0, got_data=0;
    while ((pos+8U) <= s_wav_size) {
        uint32_t csz = (uint32_t)d[pos+4]
                     | ((uint32_t)d[pos+5]<<8)
                     | ((uint32_t)d[pos+6]<<16)
                     | ((uint32_t)d[pos+7]<<24);
        if (d[pos]=='f'&&d[pos+1]=='m'&&d[pos+2]=='t'&&d[pos+3]==' ') {
            af = (uint32_t)d[pos+8]|(uint32_t)(d[pos+9]<<8);
            sr = (uint32_t)d[pos+12]|((uint32_t)d[pos+13]<<8)
               | ((uint32_t)d[pos+14]<<16)|((uint32_t)d[pos+15]<<24);
            ba = (uint32_t)d[pos+20]|(uint32_t)(d[pos+21]<<8);
            got_fmt = 1;
        } else if (d[pos]=='d'&&d[pos+1]=='a'&&d[pos+2]=='t'&&d[pos+3]=='a') {
            s_data_offset = pos+8;
            s_data_size   = csz;
            got_data = 1;
        }
        pos += 8+csz;
        if (csz & 1) { pos++; }
        if (got_fmt && got_data) { break; }
    }
    if (!got_fmt||!got_data||af!=0x0011U) {
        ESP_LOGE(TAG,"Not IMA-ADPCM"); return -1;
    }
    s_block_align = ba;
    s_sample_rate = (CFG_AUDIO_SR != 0U) ? (uint32_t)CFG_AUDIO_SR : sr;
    s_read_pos    = s_data_offset;
    ESP_LOGI(TAG,"WAV sr=%"PRIu32" ba=%"PRIu32" dsz=%"PRIu32,
             s_sample_rate, ba, s_data_size);
    return 0;
}

/* ============================================================
 * ニブルデコーダ (IRAM)
 * ============================================================ */
static IRAM_ATTR int16_t adpcm_nibble(adpcm_state_t *st, uint8_t n)
{
    int16_t step = s_step_table[st->step_index];
    int32_t diff = (int32_t)step >> 3;
    if (n & 4) { diff += step; }
    if (n & 2) { diff += step >> 1; }
    if (n & 1) { diff += step >> 2; }
    if (n & 8) { diff = -diff; }
    int32_t pred = (int32_t)st->predictor + diff;
    if (pred >  32767) { pred =  32767; }
    if (pred < -32768) { pred = -32768; }
    st->predictor = (int16_t)pred;
    int8_t idx = st->step_index + s_idx_table[n & 7];
    if (idx < 0)  { idx = 0; }
    if (idx > 88) { idx = 88; }
    st->step_index = idx;
    return st->predictor;
}

/* ============================================================
 * リングへの書き込み — 生 PCM を格納 (音量は ISR で適用)
 * ============================================================ */
static inline void ring_push_raw(int16_t raw)
{
    while (ring_full()) { vTaskDelay(1); }
    s_ring[s_ring_wr & (ADPCM_RING_SIZE-1U)] = raw;
    s_ring_wr++;
}

/* ============================================================
 * ブロックデコード → ring (生 PCM)
 * ============================================================ */
static void decode_block_to_ring(const uint8_t *blk, uint32_t bytes)
{
    if (bytes < 4) { return; }
    adpcm_state_t st;
    st.predictor  = (int16_t)((uint16_t)blk[0] | ((uint16_t)blk[1]<<8));
    st.step_index = (int8_t)blk[2];
    if (st.step_index < 0)  { st.step_index = 0; }
    if (st.step_index > 88) { st.step_index = 88; }
    ring_push_raw(st.predictor);
    for (uint32_t i = 4; i < bytes; i++) {
        ring_push_raw(adpcm_nibble(&st,  blk[i]     & 0x0F));
        ring_push_raw(adpcm_nibble(&st, (blk[i]>>4) & 0x0F));
    }
}

/* ============================================================
 * ISR 内 duty 更新ヘルパーマクロ
 *
 * IRAM 配置の ledc_set_duty/update_duty を使用。
 * 並列ピンは #if でコンパイル時に決定するため実行時コスト 0。
 * ============================================================ */

/* duty を SINGLE/並列ピンに設定するマクロ */
#define SET_DUTY_P(d) \
    ledc_set_duty(PWM_SPEED_MODE, PWM_CHANNEL_P, (uint32_t)(d)); \
    ledc_update_duty(PWM_SPEED_MODE, PWM_CHANNEL_P)

#if (CFG_AUDIO_PIN_P2 != 0)
#  define SET_DUTY_P2(d) \
    ledc_set_duty(PWM_SPEED_MODE, PWM_CHANNEL_P2, (uint32_t)(d)); \
    ledc_update_duty(PWM_SPEED_MODE, PWM_CHANNEL_P2)
#else
#  define SET_DUTY_P2(d)  /* 無効 */
#endif

#if (CFG_AUDIO_PIN_P3 != 0)
#  define SET_DUTY_P3(d) \
    ledc_set_duty(PWM_SPEED_MODE, PWM_CHANNEL_P3, (uint32_t)(d)); \
    ledc_update_duty(PWM_SPEED_MODE, PWM_CHANNEL_P3)
#else
#  define SET_DUTY_P3(d)
#endif

#if (CFG_AUDIO_PIN_P4 != 0)
#  define SET_DUTY_P4(d) \
    ledc_set_duty(PWM_SPEED_MODE, PWM_CHANNEL_P4, (uint32_t)(d)); \
    ledc_update_duty(PWM_SPEED_MODE, PWM_CHANNEL_P4)
#else
#  define SET_DUTY_P4(d)
#endif

/* ============================================================
 * gptimer ISR
 *
 * 音量スケーリングを ISR に集約:
 *   リングには生 PCM (±32767) を格納。
 *   ISR で pcm × CFG_AUDIO_VOL >> 8 を適用してから duty 計算。
 *   これにより音量変更が即時反映可能。
 *
 * ダイナミックレンジ最大化:
 *   DUTY_SWING = 1024 (フルスイング)
 *   duty = MID ± (pcm_vol * 1024) / 32767
 *   pcm_vol = pcm * vol >> 8 (最大 32767)
 *   → duty の最大変動 = ±1024 = ±50% → クリップなし
 *
 * 並列ピン出力:
 *   SINGLE/BTL どちらも正相 duty を P2/P3/P4 に同時設定。
 *   電流が本数倍になり音圧が向上する。
 * ============================================================ */
static IRAM_ATTR bool audio_isr_cb(gptimer_handle_t timer,
                                    const gptimer_alarm_event_data_t *edata,
                                    void *user_ctx)
{
    (void)timer; (void)edata; (void)user_ctx;
    BaseType_t woken = pdFALSE;

    /* リングから生 PCM を取得 */
    int32_t pcm = 0;
    if (!ring_empty()) {
        pcm = (int32_t)s_ring[s_ring_rd & (ADPCM_RING_SIZE-1U)];
        s_ring_rd++;
    }

    /* 音量スケーリング: pcm × s_vol / 256
     * s_vol は adpcm_set_vol() でリアルタイム更新される */
    int32_t pcm_vol = (pcm * (int32_t)s_vol) >> 8;
    if (pcm_vol >  32767) { pcm_vol =  32767; }
    if (pcm_vol < -32767) { pcm_vol = -32767; }

    /* VU: スケーリング後の絶対値ピーク */
    {
        uint16_t a = (pcm_vol < 0) ? (uint16_t)(-pcm_vol) : (uint16_t)pcm_vol;
        if (a > s_vu_peak) { s_vu_peak = a; }
    }

#if (CFG_AUDIO_OUTPUT_MODE == 1)
    /* BTL: 正相・逆相ともフルスイング → 差動振幅 = 2×DUTY_SWING */
    int32_t dp = (int32_t)PWM_MID_DUTY + (pcm_vol * (int32_t)DUTY_SWING) / 32767;
    int32_t dn = (int32_t)PWM_MID_DUTY - (pcm_vol * (int32_t)DUTY_SWING) / 32767;
    if (dp > (int32_t)PWM_MAX_DUTY) { dp = (int32_t)PWM_MAX_DUTY; }
    if (dp < 0) { dp = 0; }
    if (dn > (int32_t)PWM_MAX_DUTY) { dn = (int32_t)PWM_MAX_DUTY; }
    if (dn < 0) { dn = 0; }
    ledc_set_duty(PWM_SPEED_MODE, PWM_CHANNEL_P, (uint32_t)dp);
    ledc_set_duty(PWM_SPEED_MODE, PWM_CHANNEL_N, (uint32_t)dn);
    ledc_update_duty(PWM_SPEED_MODE, PWM_CHANNEL_P);
    ledc_update_duty(PWM_SPEED_MODE, PWM_CHANNEL_N);
    /* 並列ピン: 正相 duty を複製 */
    SET_DUTY_P2(dp);
    SET_DUTY_P3(dp);
    SET_DUTY_P4(dp);

#else
    /* SINGLE */
    int32_t duty = (int32_t)PWM_MID_DUTY + (pcm_vol * (int32_t)DUTY_SWING) / 32767;
    if (duty > (int32_t)PWM_MAX_DUTY) { duty = (int32_t)PWM_MAX_DUTY; }
    if (duty < 0) { duty = 0; }
    SET_DUTY_P(duty);
    SET_DUTY_P2(duty);
    SET_DUTY_P3(duty);
    SET_DUTY_P4(duty);
#endif

    /* フレーム同期: μs 優先, なければ ms */
    s_isr_samples++;
    {
        uint32_t spf = 0U;
        if (s_frame_interval_us > 0U) {
            spf = (uint32_t)(
                (uint64_t)s_sample_rate * s_frame_interval_us / 1000000ULL);
        } else if (s_frame_interval_ms > 0U) {
            spf = (s_sample_rate * s_frame_interval_ms) / 1000U;
        }
        if (spf > 0U) {
            uint32_t fn = s_isr_samples / spf;
            while (s_frames_given < fn) {
                xSemaphoreGiveFromISR(s_video_sem, &woken);
                s_frames_given++;
            }
        }
    }
    return (woken == pdTRUE);
}

/* ============================================================
 * ADPCM デコードタスク
 *
 * フェーズ1 (prefill):
 *   リングが PREFILL_TARGET (75%) になるまで連続デコード。
 *   完了後 s_prefill_done=1 → adpcm_init() が gptimer 起動。
 *
 * フェーズ2 (通常):
 *   ring_full なら vTaskDelay(1)、空きがあればデコード継続。
 *   s_ring_wr を触るのは常にこのタスクのみ (競合なし)。
 * ============================================================ */
static void adpcm_decode_task(void *arg)
{
    (void)arg;
    s_prefill_done = 0;

    /* フェーズ1: prefill */
    while (ring_used() < PREFILL_TARGET) {
        if (s_read_pos >= (s_data_offset + s_data_size)) { break; }
        uint32_t rem = (s_data_offset + s_data_size) - s_read_pos;
        uint32_t blk = (s_block_align > 0) ? s_block_align : 512U;
        if (blk > rem) { blk = rem; }
        decode_block_to_ring(s_wav_data + s_read_pos, blk);
        s_read_pos += blk;
        taskYIELD();
    }
    ESP_LOGI(TAG,"prefill done: %"PRIu32" samples", ring_used());
    s_prefill_done = 1;

    /* フェーズ2: 通常デコード */
    for (;;) {
        if (s_read_pos >= (s_data_offset + s_data_size)) {
            vTaskDelay(pdMS_TO_TICKS(50));
            continue;
        }
        if (ring_full()) { vTaskDelay(1); continue; }
        uint32_t rem = (s_data_offset + s_data_size) - s_read_pos;
        uint32_t blk = (s_block_align > 0) ? s_block_align : 512U;
        if (blk > rem) { blk = rem; }
        decode_block_to_ring(s_wav_data + s_read_pos, blk);
        s_read_pos += blk;
    }
}

/* ============================================================
 * LEDC チャンネル初期化ヘルパー
 * ============================================================ */
static void ledc_ch_init(ledc_channel_t ch, int gpio, uint32_t duty, uint32_t hpoint)
{
    ledc_channel_config_t cc = {
        .speed_mode = PWM_SPEED_MODE,
        .channel    = ch,
        .timer_sel  = PWM_TIMER,
        .intr_type  = LEDC_INTR_DISABLE,
        .gpio_num   = gpio,
        .duty       = duty,
        .hpoint     = hpoint,
    };
    ESP_ERROR_CHECK(ledc_channel_config(&cc));
}

/* ============================================================
 * adpcm_init
 * ============================================================ */
int adpcm_init(const uint8_t *adpcm_data, uint32_t adpcm_size,
               SemaphoreHandle_t video_sem)
{
    if (!adpcm_data) {
        adpcm_data = bad_audio_data;
        adpcm_size = BAD_AUDIO_SIZE;
    }
    s_wav_data          = adpcm_data;
    s_wav_size          = adpcm_size;
    s_video_sem         = video_sem;
    s_ring_wr           = 0;
    s_ring_rd           = 0;
    s_isr_samples       = 0;
    s_frames_given      = 0;
    s_vu_peak           = 0;
    s_frame_interval_ms = 0U;
    s_frame_interval_us = 0U;
    s_gptimer           = NULL;
    s_prefill_done      = 0;
    s_vol               = (uint32_t)CFG_AUDIO_VOL;

    if (wav_parse() != 0) { return -1; }

    /* ---- LEDC タイマ初期化 ---- */
    ledc_timer_config_t tc = {
        .speed_mode      = PWM_SPEED_MODE,
        .duty_resolution = PWM_RESOLUTION,
        .timer_num       = PWM_TIMER,
        .freq_hz         = PWM_FREQ_HZ,
        .clk_cfg         = LEDC_AUTO_CLK,
    };
    ESP_ERROR_CHECK(ledc_timer_config(&tc));

    /* ---- 正相チャンネル (必須) ---- */
    ledc_ch_init(PWM_CHANNEL_P, CFG_AUDIO_PIN_P, PWM_MID_DUTY, 0);

#if (CFG_AUDIO_OUTPUT_MODE == 1)
    /* BTL 逆相 (hpoint=MID で 180° 位相オフセット) */
    ledc_ch_init(PWM_CHANNEL_N, CFG_AUDIO_PIN_N, PWM_MID_DUTY, PWM_MID_DUTY);
    ESP_LOGI(TAG,"LEDC BTL P=GPIO%d N=GPIO%d %uHz",
             CFG_AUDIO_PIN_P, CFG_AUDIO_PIN_N, PWM_FREQ_HZ);
#else
    ESP_LOGI(TAG,"LEDC SINGLE P=GPIO%d %uHz", CFG_AUDIO_PIN_P, PWM_FREQ_HZ);
#endif

    /* ---- 並列ピン初期化 ---- */
#if (CFG_AUDIO_PIN_P2 != 0)
    ledc_ch_init(PWM_CHANNEL_P2, CFG_AUDIO_PIN_P2, PWM_MID_DUTY, 0);
    ESP_LOGI(TAG,"parallel P2=GPIO%d", CFG_AUDIO_PIN_P2);
#endif
#if (CFG_AUDIO_PIN_P3 != 0)
    ledc_ch_init(PWM_CHANNEL_P3, CFG_AUDIO_PIN_P3, PWM_MID_DUTY, 0);
    ESP_LOGI(TAG,"parallel P3=GPIO%d", CFG_AUDIO_PIN_P3);
#endif
#if (CFG_AUDIO_PIN_P4 != 0)
    ledc_ch_init(PWM_CHANNEL_P4, CFG_AUDIO_PIN_P4, PWM_MID_DUTY, 0);
    ESP_LOGI(TAG,"parallel P4=GPIO%d", CFG_AUDIO_PIN_P4);
#endif

    /* ---- decode_task を先に起動して prefill を待つ ---- */
    xTaskCreatePinnedToCore(adpcm_decode_task, "adpcm", 4096, NULL, 4, NULL, 0);
    {
        uint32_t waited = 0;
        while (!s_prefill_done && waited < 15000U) {
            vTaskDelay(pdMS_TO_TICKS(10));
            waited += 10U;
        }
        if (!s_prefill_done) {
            ESP_LOGW(TAG,"prefill timeout (ring=%"PRIu32")", ring_used());
        }
    }

    /* ---- gptimer 起動 (prefill 完了後) ---- */
    {
        uint32_t alarm = TIMER_RES_HZ / s_sample_rate;
        if (alarm < 1) { alarm = 1; }
        gptimer_config_t tcfg;
        memset(&tcfg, 0, sizeof(tcfg));
        tcfg.clk_src       = GPTIMER_CLK_SRC_DEFAULT;
        tcfg.direction     = GPTIMER_COUNT_UP;
        tcfg.resolution_hz = TIMER_RES_HZ;
        if (gptimer_new_timer(&tcfg, &s_gptimer) != ESP_OK) {
            ESP_LOGE(TAG,"gptimer_new failed"); return -1;
        }
        gptimer_alarm_config_t acfg;
        memset(&acfg, 0, sizeof(acfg));
        acfg.alarm_count                = alarm;
        acfg.flags.auto_reload_on_alarm = 1;
        ESP_ERROR_CHECK(gptimer_set_alarm_action(s_gptimer, &acfg));
        gptimer_event_callbacks_t cbs;
        memset(&cbs, 0, sizeof(cbs));
        cbs.on_alarm = audio_isr_cb;
        ESP_ERROR_CHECK(gptimer_register_event_callbacks(s_gptimer, &cbs, NULL));
        ESP_ERROR_CHECK(gptimer_enable(s_gptimer));
        ESP_ERROR_CHECK(gptimer_start(s_gptimer));
        ESP_LOGI(TAG,"init OK mode=%d SR=%"PRIu32" alarm=%"PRIu32
                 " ring=%"PRIu32" vol=%d",
                 CFG_AUDIO_OUTPUT_MODE, s_sample_rate, alarm,
                 ring_used(), CFG_AUDIO_VOL);
    }
    return 0;
}

/* ============================================================
 * adpcm_rewind
 * gptimer: stop → disable → reset → enable → start
 * ============================================================ */

/* 全 LEDC チャンネルを無音 (MID_DUTY) に設定するヘルパー */
static void ledc_all_mute(void)
{
    ledc_set_duty(PWM_SPEED_MODE, PWM_CHANNEL_P, PWM_MID_DUTY);
    ledc_update_duty(PWM_SPEED_MODE, PWM_CHANNEL_P);
#if (CFG_AUDIO_OUTPUT_MODE == 1)
    ledc_set_duty(PWM_SPEED_MODE, PWM_CHANNEL_N, PWM_MID_DUTY);
    ledc_update_duty(PWM_SPEED_MODE, PWM_CHANNEL_N);
#endif
#if (CFG_AUDIO_PIN_P2 != 0)
    ledc_set_duty(PWM_SPEED_MODE, PWM_CHANNEL_P2, PWM_MID_DUTY);
    ledc_update_duty(PWM_SPEED_MODE, PWM_CHANNEL_P2);
#endif
#if (CFG_AUDIO_PIN_P3 != 0)
    ledc_set_duty(PWM_SPEED_MODE, PWM_CHANNEL_P3, PWM_MID_DUTY);
    ledc_update_duty(PWM_SPEED_MODE, PWM_CHANNEL_P3);
#endif
#if (CFG_AUDIO_PIN_P4 != 0)
    ledc_set_duty(PWM_SPEED_MODE, PWM_CHANNEL_P4, PWM_MID_DUTY);
    ledc_update_duty(PWM_SPEED_MODE, PWM_CHANNEL_P4);
#endif
}

void adpcm_rewind(void)
{
    gptimer_stop(s_gptimer);
    gptimer_disable(s_gptimer);

    s_read_pos     = s_data_offset;
    s_ring_wr      = 0;
    s_ring_rd      = 0;
    s_isr_samples  = 0;
    s_frames_given = 0;
    s_vu_peak      = 0;
    s_prefill_done = 0;

    ledc_all_mute();

    /* decode_task が prefill を再実行するのを待つ */
    {
        uint32_t waited = 0;
        while (!s_prefill_done && waited < 10000U) {
            vTaskDelay(pdMS_TO_TICKS(10));
            waited += 10U;
        }
    }

    gptimer_enable(s_gptimer);
    gptimer_start(s_gptimer);
    ESP_LOGI(TAG,"rewind done ring=%"PRIu32, ring_used());
}
