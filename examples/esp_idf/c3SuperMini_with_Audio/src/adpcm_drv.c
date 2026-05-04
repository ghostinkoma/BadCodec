/**
 * @file  adpcm_drv.c
 * @brief IMA ADPCM decoder + LEDC PWM / HW SDM audio output
 * @version v3.0.0
 *
 * ============================================================
 * ESP-IDF v5.0.2 / ESP32-C3 完全対応版
 * ============================================================
 *
 * 出力モード (CFG_AUDIO_OUTPUT_MODE):
 *   0 = SINGLE  シングルエンド LEDC PWM  (GPIO_P のみ)
 *   1 = BTL     差動 LEDC PWM            (GPIO_P 正相 + GPIO_N 逆相)
 *   2 = SDM     ハードウェア SDM         (GPIO_P のみ)
 *
 * --- mode 0/1 信号経路 ---
 *   WAV → ring → gptimer ISR → ledc_set_duty() → GPIO PWM
 *   ledc_set_duty() は IRAM 配置 → ISR から安全に呼び出し可能
 *
 * --- mode 2 信号経路 ---
 *   WAV → ring → gptimer ISR → vTaskNotifyGiveFromISR()
 *              → sdm_task (タスクコンテキスト) → sdm_channel_set_duty()
 *   sdm_channel_set_duty() は Flash 配置 → ISR 不可 → TaskNotify 経由
 *   sdm_task は ulTaskNotifyTake(portMAX_DELAY) でブロック待機するため
 *   CPU を独占せず、アイドルタスクが確実に動く → WDT タイムアウトなし
 *
 * --- include 設計 ---
 *   adpcm_drv.h は config.h を include しない (循環防止)。
 *   config.h は本ファイルのみが include する。
 *
 * ============================================================
 * 修正履歴
 * ============================================================
 * v3.0.0:
 *   - I2S PDM 方式を廃止し LEDC/SDM + gptimer 方式に統一
 *     (I2S PDM は up_sample_fp/fs 設定が IDF v5.0.2 で正確に動作しない)
 *   - CFG_AUDIO_VOL=5→220 に修正 (5では S/N壊滅)
 *   - BTL: 両チャンネルを DUTY_SWING フルスイングに修正 (旧版は半振幅)
 *   - SDM mode=2: gptimer ISR → TaskNotify → sdm_task の安全な構成
 *   - adpcm_drv.h から config.h include を削除 (循環 include 解消)
 *   - prefill_ring() を共通関数化 (init/rewind で共用)
 *   - WDT: prefill 中に vTaskDelay(1) で回避 (esp_task_wdt_reset() は使わない)
 */

/* config.h を最初に include (CFG_AUDIO_OUTPUT_MODE 等を定義) */
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

#if (CFG_AUDIO_OUTPUT_MODE == 2)
#  include "driver/sdm.h"
#endif

static const char *const TAG = "ADPCM";

/* ---- LEDC 設定 (mode 0/1) -------------------------------- */
#define PWM_RESOLUTION   LEDC_TIMER_11_BIT  /* 2048段階 */
#define PWM_MAX_DUTY     2047U
#define PWM_MID_DUTY     1024U              /* 無音=中点 */
#define PWM_FREQ_HZ      39062U             /* ~39kHz キャリア */
#define PWM_TIMER        LEDC_TIMER_0
#define PWM_SPEED_MODE   LEDC_LOW_SPEED_MODE
#define PWM_CHANNEL_P    LEDC_CHANNEL_0     /* 正相 */
#define PWM_CHANNEL_N    LEDC_CHANNEL_1     /* 逆相 (BTL) */
/* フルスイング: PCM±32767 → duty ±DUTY_SWING (中点から) */
#define DUTY_SWING       1002U

/* ---- gptimer 設定 ---------------------------------------- */
#define TIMER_RES_HZ     1000000U           /* 1μs 分解能 */

/* ---- IMA ADPCM テーブル ----------------------------------- */
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

/* ---- リングバッファ --------------------------------------- */
static volatile int16_t  s_ring[ADPCM_RING_SIZE];
static volatile uint32_t s_ring_wr;
static volatile uint32_t s_ring_rd;
static inline uint32_t ring_used(void)
{ return (s_ring_wr - s_ring_rd) & (ADPCM_RING_SIZE-1U); }
static inline int ring_full(void)
{ return ring_used() >= (ADPCM_RING_SIZE-1U); }
static inline int ring_empty(void)
{ return s_ring_wr == s_ring_rd; }

/* ---- 状態変数 --------------------------------------------- */
static const uint8_t    *s_wav_data;
static uint32_t          s_wav_size;
static uint32_t          s_data_offset;
static uint32_t          s_data_size;
static uint32_t          s_block_align;
static uint32_t          s_sample_rate;
static uint32_t          s_read_pos;
static SemaphoreHandle_t s_video_sem;
static gptimer_handle_t  s_gptimer;
static volatile uint32_t s_frame_interval_ms; /* ms単位 (後方互換) */
static volatile uint32_t s_frame_interval_us; /* μs単位 (高精度) */
/* 0 でない方を優先使用: us > ms */

static volatile uint32_t s_isr_samples;
static volatile uint32_t s_frames_given;
static volatile uint16_t s_vu_peak;

#if (CFG_AUDIO_OUTPUT_MODE == 2)
static sdm_channel_handle_t s_sdm_ch;
static TaskHandle_t          s_sdm_task_hdl;
#endif

/* ---- Public API ------------------------------------------- */
void     adpcm_set_frame_ms(uint32_t ms) { s_frame_interval_ms = ms; s_frame_interval_us = 0U; }
void     adpcm_set_frame_us(uint32_t us) { s_frame_interval_us = us; s_frame_interval_ms = 0U; }
uint32_t adpcm_get_sample_rate(void)     { return s_sample_rate; }
uint16_t adpcm_get_vu(void)
{
    uint16_t v = s_vu_peak;
    s_vu_peak  = (uint16_t)(s_vu_peak - (s_vu_peak >> 3)); /* 緩減衰 */
    return v;
}

/* ==========================================================
 * WAV パーサ
 * ==========================================================
 * サンプルレート優先順位:
 *   1. CFG_AUDIO_SR != 0 → 強制値
 *   2. BAD_AUDIO_SAMPLE_RATE (adpcm4.h / Wave2adpcmH.py が生成)
 *      .bad 映像と同じエンコーダで生成した正確な値を使う。
 * ========================================================== */
static int wav_parse(void)
{
    const uint8_t *d = s_wav_data;
    if (s_wav_size < 44U)
    { ESP_LOGE(TAG,"WAV too small"); return -1; }
    if (d[0]!='R'||d[1]!='I'||d[2]!='F'||d[3]!='F')
    { ESP_LOGE(TAG,"No RIFF"); return -1; }
    if (d[8]!='W'||d[9]!='A'||d[10]!='V'||d[11]!='E')
    { ESP_LOGE(TAG,"No WAVE"); return -1; }

    uint32_t pos=12, af=0, sr_wav=0, ba=0;
    int got_fmt=0, got_data=0;
    while ((pos+8U) <= s_wav_size) {
        uint32_t csz = (uint32_t)d[pos+4] | ((uint32_t)d[pos+5]<<8)
                     | ((uint32_t)d[pos+6]<<16) | ((uint32_t)d[pos+7]<<24);
        if (d[pos]=='f'&&d[pos+1]=='m'&&d[pos+2]=='t'&&d[pos+3]==' ') {
            af     = (uint32_t)d[pos+8]  | ((uint32_t)d[pos+9]<<8);
            sr_wav = (uint32_t)d[pos+12] | ((uint32_t)d[pos+13]<<8)
                   | ((uint32_t)d[pos+14]<<16) | ((uint32_t)d[pos+15]<<24);
            ba     = (uint32_t)d[pos+20] | ((uint32_t)d[pos+21]<<8);
            got_fmt = 1;
        } else if (d[pos]=='d'&&d[pos+1]=='a'&&d[pos+2]=='t'&&d[pos+3]=='a') {
            s_data_offset = pos+8;
            s_data_size   = csz;
            got_data = 1;
        }
        pos += 8+csz;
        if (csz & 1) { pos++; }          /* RIFF パディング */
        if (got_fmt && got_data) { break; }
    }
    if (!got_fmt || !got_data)
    { ESP_LOGE(TAG,"chunks missing"); return -1; }
    if (af != 0x0011U)
    { ESP_LOGE(TAG,"Not IMA-ADPCM (fmt=0x%04"PRIx32")",af); return -1; }

    s_block_align = ba;
    s_sample_rate = (CFG_AUDIO_SR != 0U)
                    ? (uint32_t)CFG_AUDIO_SR
                    : (uint32_t)BAD_AUDIO_SAMPLE_RATE;
    s_read_pos    = s_data_offset;

    ESP_LOGI(TAG,"WAV OK sr_h=%"PRIu32" sr_use=%"PRIu32" ba=%"PRIu32
             " mode=%d vol=%d",
             sr_wav, s_sample_rate, ba,
             CFG_AUDIO_OUTPUT_MODE, CFG_AUDIO_VOL);
    return 0;
}

/* ==========================================================
 * IMA ADPCM ニブルデコーダ
 * ========================================================== */
static int16_t adpcm_nibble(adpcm_state_t *st, uint8_t n)
{
    int16_t step = s_step_table[st->step_index];
    int32_t diff = (int32_t)step >> 3;
    if (n&4){ diff += step; }
    if (n&2){ diff += step>>1; }
    if (n&1){ diff += step>>2; }
    if (n&8){ diff = -diff; }
    int32_t pred = (int32_t)st->predictor + diff;
    if (pred >  32767){ pred =  32767; }
    if (pred < -32768){ pred = -32768; }
    st->predictor = (int16_t)pred;
    int8_t idx = st->step_index + s_idx_table[n&7];
    if (idx < 0){ idx=0; } else if (idx > 88){ idx=88; }
    st->step_index = idx;
    return st->predictor;
}

/* ==========================================================
 * PCM サンプルを ring に書き込む (音量適用)
 *
 * CFG_AUDIO_VOL: 0=無音, 256=フルスケール
 *   推奨値: 220 (歪みなし・S/N最良)
 *   BTL接続では両ピンがフルスイングするため
 *   VOL=220 でも実効差動振幅は十分大きい。
 * ========================================================== */
static inline void ring_push(int16_t raw)
{
    int32_t v = ((int32_t)raw * (int32_t)CFG_AUDIO_VOL) >> 8;
    if (v >  32767){ v =  32767; }
    if (v < -32768){ v = -32768; }
    while (ring_full()) { vTaskDelay(1); }
    s_ring[s_ring_wr & (ADPCM_RING_SIZE-1U)] = (int16_t)v;
    s_ring_wr++;
}

/* ==========================================================
 * ブロックデコード → ring
 * ========================================================== */
static void decode_block_to_ring(const uint8_t *blk, uint32_t bytes)
{
    if (bytes < 4) return;
    adpcm_state_t st;
    st.predictor  = (int16_t)((uint16_t)blk[0]|((uint16_t)blk[1]<<8));
    st.step_index = (int8_t)blk[2];
    if (st.step_index < 0){ st.step_index=0; }
    if (st.step_index > 88){ st.step_index=88; }
    ring_push(st.predictor);
    for (uint32_t i=4; i<bytes; i++) {
        ring_push(adpcm_nibble(&st,  blk[i]      & 0x0F));
        ring_push(adpcm_nibble(&st, (blk[i]>>4)  & 0x0F));
    }
}

/* ==========================================================
 * ring 事前充填 (共通サブルーチン)
 * vTaskDelay(1) 使用: WDT 安全、スケジューラへ制御を返す
 * ========================================================== */
static void prefill_ring(void)
{
    uint32_t filled = 0;
    while (filled < (ADPCM_RING_SIZE-1U)) {
        if (s_read_pos >= (s_data_offset + s_data_size)) break;
        uint32_t rem = (s_data_offset + s_data_size) - s_read_pos;
        uint32_t blk = (s_block_align > 0) ? s_block_align : 512U;
        if (blk > rem) { blk = rem; }
        const uint8_t *p = s_wav_data + s_read_pos;
        adpcm_state_t st;
        st.predictor  = (int16_t)((uint16_t)p[0]|((uint16_t)p[1]<<8));
        st.step_index = (int8_t)p[2];
        if (st.step_index<0){ st.step_index=0; }
        if (st.step_index>88){ st.step_index=88; }
        /* ブロック先頭サンプル */
        if (!ring_full()) {
            int32_t v=((int32_t)st.predictor*(int32_t)CFG_AUDIO_VOL)>>8;
            if(v> 32767){v= 32767;} if(v<-32768){v=-32768;}
            s_ring[s_ring_wr&(ADPCM_RING_SIZE-1U)]=(int16_t)v;
            s_ring_wr++; filled++;
        }
        /* ニブル展開 */
        for (uint32_t i=4; i<blk && !ring_full(); i++) {
            int16_t s0=adpcm_nibble(&st, p[i]     &0x0F);
            int16_t s1=adpcm_nibble(&st,(p[i]>>4) &0x0F);
            int32_t v0=((int32_t)s0*(int32_t)CFG_AUDIO_VOL)>>8;
            int32_t v1=((int32_t)s1*(int32_t)CFG_AUDIO_VOL)>>8;
            if(v0> 32767){v0= 32767;} if(v0<-32768){v0=-32768;}
            if(v1> 32767){v1= 32767;} if(v1<-32768){v1=-32768;}
            s_ring[s_ring_wr&(ADPCM_RING_SIZE-1U)]=(int16_t)v0;
            s_ring_wr++; filled++;
            if (ring_full()) break;
            s_ring[s_ring_wr&(ADPCM_RING_SIZE-1U)]=(int16_t)v1;
            s_ring_wr++; filled++;
        }
        s_read_pos += blk;
        /* 512 サンプルごとに vTaskDelay(1) → アイドルタスクが動く */
        if ((filled & 511U) == 0U) { vTaskDelay(1); }
    }
    ESP_LOGI(TAG,"prefill: %"PRIu32" samples", filled);
}

/* ==========================================================
 * gptimer ISR — 全モード共通
 *
 * mode 0/1 (LEDC):
 *   ledc_set_duty() は IRAM 配置 → ISR から直接呼び出し可能。
 *
 * mode 2 (SDM):
 *   sdm_channel_set_duty() は Flash 配置 → ISR 不可。
 *   vTaskNotifyGiveFromISR() で sdm_task を起こす。
 *   sdm_task がタスクコンテキストで sdm_channel_set_duty() を呼ぶ。
 *
 * フレーム同期: 消費サンプル数から映像タイミングを計算。
 * ========================================================== */
static IRAM_ATTR bool audio_isr_cb(gptimer_handle_t timer,
                                    const gptimer_alarm_event_data_t *edata,
                                    void *user_ctx)
{
    (void)timer; (void)edata; (void)user_ctx;
    BaseType_t woken = pdFALSE;

#if (CFG_AUDIO_OUTPUT_MODE == 2)
    /* SDM: TaskNotify で sdm_task を起こす */
    vTaskNotifyGiveFromISR(s_sdm_task_hdl, &woken);

#else
    /* LEDC mode 0/1: ISR 内で直接 duty 更新 */
    int32_t pcm = 0;
    if (!ring_empty()) {
        pcm = (int32_t)s_ring[s_ring_rd & (ADPCM_RING_SIZE-1U)];
        s_ring_rd++;
    }
    /* VU */
    { uint16_t a=(pcm<0)?(uint16_t)(-pcm):(uint16_t)pcm;
      if(a>s_vu_peak){s_vu_peak=a;} }

#  if (CFG_AUDIO_OUTPUT_MODE == 1)
    /* BTL: 正相・逆相ともフルスイング → 差動振幅 = 2×DUTY_SWING */
    int32_t dp=(int32_t)PWM_MID_DUTY+(pcm*(int32_t)DUTY_SWING)/32767;
    int32_t dn=(int32_t)PWM_MID_DUTY-(pcm*(int32_t)DUTY_SWING)/32767;
    if(dp>(int32_t)PWM_MAX_DUTY){dp=(int32_t)PWM_MAX_DUTY;}
    if(dp<0){dp=0;}
    if(dn>(int32_t)PWM_MAX_DUTY){dn=(int32_t)PWM_MAX_DUTY;}
    if(dn<0){dn=0;}
    ledc_set_duty(PWM_SPEED_MODE,PWM_CHANNEL_P,(uint32_t)dp);
    ledc_set_duty(PWM_SPEED_MODE,PWM_CHANNEL_N,(uint32_t)dn);
    ledc_update_duty(PWM_SPEED_MODE,PWM_CHANNEL_P);
    ledc_update_duty(PWM_SPEED_MODE,PWM_CHANNEL_N);
#  else
    /* SINGLE */
    int32_t duty=(int32_t)PWM_MID_DUTY+(pcm*(int32_t)DUTY_SWING)/32767;
    if(duty>(int32_t)PWM_MAX_DUTY){duty=(int32_t)PWM_MAX_DUTY;}
    if(duty<0){duty=0;}
    ledc_set_duty(PWM_SPEED_MODE,PWM_CHANNEL_P,(uint32_t)duty);
    ledc_update_duty(PWM_SPEED_MODE,PWM_CHANNEL_P);
#  endif
#endif /* CFG_AUDIO_OUTPUT_MODE */

    /* フレーム同期 */
    s_isr_samples++;
    /* フレーム同期: μs 優先 (高精度)、なければ ms を使用 */
    {
        uint32_t spf = 0U;
        if (s_frame_interval_us > 0U) {
            /* μs 単位: spf = sr * interval_us / 1000000 */
            spf = (uint32_t)(
                (uint64_t)s_sample_rate * s_frame_interval_us
                / 1000000ULL);
        } else if (s_frame_interval_ms > 0U) {
            /* ms 単位: spf = sr * interval_ms / 1000 */
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

/* ==========================================================
 * SDM タスク (mode=2 専用)
 *
 * gptimer ISR から TaskNotify で起こされるたびに
 * ring から 1 サンプルを取り出して sdm_channel_set_duty() に渡す。
 * ulTaskNotifyTake(portMAX_DELAY) でブロック → CPU を独占しない
 * → アイドルタスクが確実に動く → WDT タイムアウトなし。
 * ========================================================== */
#if (CFG_AUDIO_OUTPUT_MODE == 2)
static void sdm_task(void *arg)
{
    (void)arg;
    ESP_LOGI(TAG,"sdm_task start");
    for (;;) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        int32_t pcm = 0;
        if (!ring_empty()) {
            pcm = (int32_t)s_ring[s_ring_rd & (ADPCM_RING_SIZE-1U)];
            s_ring_rd++;
        }
        { uint16_t a=(pcm<0)?(uint16_t)(-pcm):(uint16_t)pcm;
          if(a>s_vu_peak){s_vu_peak=a;} }
        /* タスクコンテキスト → Flash キャッシュ有効 → 安全 */
        sdm_channel_set_duty(s_sdm_ch, (int8_t)(pcm>>8));
    }
}
#endif

/* ==========================================================
 * ADPCM デコードタスク
 * ========================================================== */
static void adpcm_decode_task(void *arg)
{
    (void)arg;
    for (;;) {
        if (s_read_pos >= (s_data_offset + s_data_size)) {
            vTaskDelay(pdMS_TO_TICKS(50));
            continue;
        }
        if (ring_full()) { vTaskDelay(1); continue; }
        uint32_t rem=(s_data_offset+s_data_size)-s_read_pos;
        uint32_t blk=(s_block_align>0)?s_block_align:512U;
        if (blk>rem){ blk=rem; }
        decode_block_to_ring(s_wav_data+s_read_pos, blk);
        s_read_pos += blk;
    }
}

/* ==========================================================
 * gptimer 初期化ヘルパー
 * stop→disable→enable→start の正しいライフサイクルで初期化。
 * ========================================================== */
static int gptimer_init(void)
{
    uint32_t alarm = TIMER_RES_HZ / s_sample_rate;
    if (alarm < 1) { alarm = 1; }

    gptimer_config_t tcfg;
    memset(&tcfg, 0, sizeof(tcfg));
    tcfg.clk_src       = GPTIMER_CLK_SRC_DEFAULT;
    tcfg.direction     = GPTIMER_COUNT_UP;
    tcfg.resolution_hz = TIMER_RES_HZ;
    esp_err_t e = gptimer_new_timer(&tcfg, &s_gptimer);
    if (e != ESP_OK) { ESP_LOGE(TAG,"gptimer_new: 0x%x",e); return -1; }

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

    ESP_LOGI(TAG,"gptimer alarm=%"PRIu32" (SR=%"PRIu32"Hz)", alarm, s_sample_rate);
    return 0;
}

/* ==========================================================
 * adpcm_init
 * ========================================================== */
int adpcm_init(const uint8_t *adpcm_data, uint32_t adpcm_size,
               SemaphoreHandle_t video_sem)
{
    if (!adpcm_data) {
        adpcm_data = bad_audio_data;
        adpcm_size = BAD_AUDIO_SIZE;
    }
    s_wav_data   = adpcm_data;
    s_wav_size   = adpcm_size;
    s_video_sem  = video_sem;
    s_ring_wr    = s_ring_rd    = 0;
    s_isr_samples= s_frames_given = 0;
    s_vu_peak    = 0;
    s_frame_interval_ms = 0U;
    s_frame_interval_us = 0U;
    s_gptimer    = NULL;
#if (CFG_AUDIO_OUTPUT_MODE == 2)
    s_sdm_ch     = NULL;
    s_sdm_task_hdl = NULL;
#endif

    if (wav_parse() != 0) return -1;

    /* ---- 出力ペリフェラル初期化 ---- */
#if (CFG_AUDIO_OUTPUT_MODE == 2)
    /* SDM */
    sdm_config_t scfg = {
        .gpio_num = CFG_AUDIO_PIN_P,
        .clk_src  = SDM_CLK_SRC_DEFAULT,
    };
    esp_err_t e = sdm_new_channel(&scfg, &s_sdm_ch);
    if (e != ESP_OK) { ESP_LOGE(TAG,"sdm_new: 0x%x",e); return -1; }
    e = sdm_channel_enable(s_sdm_ch);
    if (e != ESP_OK) { ESP_LOGE(TAG,"sdm_en: 0x%x",e); return -1; }
    sdm_channel_set_duty(s_sdm_ch, 0);
    ESP_LOGI(TAG,"SDM GPIO%d", CFG_AUDIO_PIN_P);

    /* sdm_task を先に起動してハンドルを取得 (ISR 登録前に必要) */
    xTaskCreatePinnedToCore(sdm_task, "sdm", 2048, NULL,
                            configMAX_PRIORITIES-1,
                            (TaskHandle_t*)&s_sdm_task_hdl, 0);

#else
    /* LEDC */
    ledc_timer_config_t tc = {
        .speed_mode      = PWM_SPEED_MODE,
        .duty_resolution = PWM_RESOLUTION,
        .timer_num       = PWM_TIMER,
        .freq_hz         = PWM_FREQ_HZ,
        .clk_cfg         = LEDC_AUTO_CLK,
    };
    ESP_ERROR_CHECK(ledc_timer_config(&tc));

    ledc_channel_config_t ccp = {
        .speed_mode = PWM_SPEED_MODE, .channel   = PWM_CHANNEL_P,
        .timer_sel  = PWM_TIMER,      .intr_type  = LEDC_INTR_DISABLE,
        .gpio_num   = CFG_AUDIO_PIN_P,.duty       = PWM_MID_DUTY,
        .hpoint     = 0,
    };
    ESP_ERROR_CHECK(ledc_channel_config(&ccp));

#  if (CFG_AUDIO_OUTPUT_MODE == 1)
    /* BTL: 逆相チャンネル (hpoint=PWM_MID_DUTY で 180° 位相ずれ) */
    ledc_channel_config_t ccn = {
        .speed_mode = PWM_SPEED_MODE, .channel   = PWM_CHANNEL_N,
        .timer_sel  = PWM_TIMER,      .intr_type  = LEDC_INTR_DISABLE,
        .gpio_num   = CFG_AUDIO_PIN_N,.duty       = PWM_MID_DUTY,
        .hpoint     = PWM_MID_DUTY,   /* 逆相オフセット */
    };
    ESP_ERROR_CHECK(ledc_channel_config(&ccn));
    ESP_LOGI(TAG,"LEDC BTL P=GPIO%d N=GPIO%d fc=%uHz",
             CFG_AUDIO_PIN_P, CFG_AUDIO_PIN_N, PWM_FREQ_HZ);
#  else
    ESP_LOGI(TAG,"LEDC SINGLE P=GPIO%d fc=%uHz", CFG_AUDIO_PIN_P, PWM_FREQ_HZ);
#  endif
#endif /* CFG_AUDIO_OUTPUT_MODE */

    /* ---- ring 事前充填 ---- */
    prefill_ring();

    /* ---- gptimer 起動 ---- */
    if (gptimer_init() != 0) return -1;

    /* ---- デコードタスク起動 ---- */
    xTaskCreatePinnedToCore(adpcm_decode_task,"adpcm",4096,NULL,4,NULL,0);

    ESP_LOGI(TAG,"adpcm_init OK mode=%d SR=%"PRIu32,
             CFG_AUDIO_OUTPUT_MODE, s_sample_rate);
    return 0;
}

/* ==========================================================
 * adpcm_rewind — 先頭から再生再開
 *
 * 音声・映像双方を 0 秒から同期再スタートさせる。
 *   1. gptimer を止める
 *   2. ring / カウンタをリセット
 *   3. 出力を無音に戻す
 *   4. ring を再充填
 *   5. gptimer を再起動
 * ========================================================== */
void adpcm_rewind(void)
{
    /* gptimer 停止 (IDF v5.0.x: stop→disable が必須) */
    gptimer_stop(s_gptimer);
    gptimer_disable(s_gptimer);

    /* カウンタリセット */
    s_read_pos    = s_data_offset;
    s_ring_wr     = s_ring_rd     = 0;
    s_isr_samples = s_frames_given = 0;
    s_vu_peak     = 0;

    /* 出力を無音に */
#if (CFG_AUDIO_OUTPUT_MODE == 2)
    if (s_sdm_ch) { sdm_channel_set_duty(s_sdm_ch, 0); }
#else
    ledc_set_duty(PWM_SPEED_MODE, PWM_CHANNEL_P, PWM_MID_DUTY);
    ledc_update_duty(PWM_SPEED_MODE, PWM_CHANNEL_P);
#  if (CFG_AUDIO_OUTPUT_MODE == 1)
    ledc_set_duty(PWM_SPEED_MODE, PWM_CHANNEL_N, PWM_MID_DUTY);
    ledc_update_duty(PWM_SPEED_MODE, PWM_CHANNEL_N);
#  endif
#endif

    /* ring 再充填 */
    prefill_ring();

    /* gptimer 再起動 */
    gptimer_enable(s_gptimer);
    gptimer_start(s_gptimer);

    ESP_LOGI(TAG,"rewind done");
}
