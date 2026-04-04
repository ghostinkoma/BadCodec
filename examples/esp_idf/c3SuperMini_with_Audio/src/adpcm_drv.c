/**
 * @file  adpcm_drv.c
 * @brief IMA ADPCM decoder + LEDC Hardware PWM audio output
 * @version v2.0.0
 *
 * ============================================================
 * アーキテクチャ刷新: ソフトウェアΣΔ → LEDC ハードウェア PWM
 * ============================================================
 *
 * 【これまでの問題の根本原因】
 *
 *   ソフトウェアΣΔ + GPIO 方式の本質的な欠陥:
 *
 *   1. ISR内ループ (v1.4.0) は「等価508kHz」ではない。
 *      GPIO の切り替えは1つの ISR コンテキスト内で連続して起きる。
 *      LPF の RC 定数 (≈10μs) から見ると、16kHz 周期の中で
 *      まとめて高速スイッチングが起きた後に長い無音区間が続く
 *      → LPF の出力は「16kHz のバースト積分」になり、
 *        ΣΔのノイズシェーピング特性が完全に崩壊する。
 *
 *   2. GPIO スイッチングは CPU が直接制御するため FreeRTOS の
 *      タスクスケジューリングと競合する。シングルコアの
 *      ESP32-C3 では I2C 転送中の割り込み処理が映像 FPS を圧迫。
 *
 *   3. ΣΔのフィードバックスケールやクランプ値の誤設定が
 *      重なり S/N 比が実用レベルに達しなかった。
 *
 * 【LEDC PWM 方式の優位性】
 *
 *   LEDC (LED Control) ペリフェラルは汎用 PWM として使用可能。
 *
 *   ・キャリア生成はハードウェア完全自律
 *     → CPU を一切使わずに PWM を継続出力
 *     → FPS への影響ゼロ
 *
 *   ・キャリア周波数 = 39,062 Hz (> 20kHz = 人間の可聴域外)
 *     → キャリアノイズが原理的に聞こえない
 *     計算: 80MHz / 2048(分解能11bit) = 39,062 Hz
 *
 *   ・duty 分解能 11bit = 2048 段階
 *     → PCM の 16bit を 11bit に右シフト(>>5)してマッピング
 *     → ダイナミックレンジ: 20×log10(2048) ≈ 66 dB
 *
 *   ・duty 更新は gptimer ISR で 16kHz に同期
 *     → ledc_set_duty() + ledc_update_duty() の2命令のみ
 *     → ISR 処理時間: ~50 cycles → 占有率 < 0.1%
 *
 * 【接続】
 *
 *   シングルエンド (推奨):
 *     GPIO10 → R=10kΩ → ┬── C=100nF ── GND   (LPF fc≈160Hz)
 *                        └── C=10μF  ── アンプ入力 (カップリング)
 *
 *   PWM LPF 設計:
 *     キャリア 39kHz を十分に減衰させるため RC を大きくする。
 *     fc=160Hz (-3dB) とすれば 39kHz では -48dB の減衰。
 *     音声帯域 (20Hz-8kHz) は通過し、キャリアは完全除去。
 *
 *     より高品質な 2段 LPF:
 *       1段目: R=10kΩ, C=100nF (fc≈160Hz)
 *       2段目: R=10kΩ, C=100nF (fc≈160Hz)
 *       → 39kHz で -96dB 減衰 (実質無音)
 *
 *   ※ ΣΔ時代の R=1kΩ, C=10nF (fc=16kHz) では PWM キャリアが
 *     通過してしまうので必ず変更すること。
 *
 * 【パラメータ】
 *
 *   PWM キャリア: 39,062 Hz  (80MHz / 2^11)
 *   PWM 分解能:   11 bit     (0〜2047)
 *   サンプルレート: 16,000 Hz (WAV ヘッダ値を使用)
 *   ISR 発火:     16,000 Hz  (1MHz タイマ、alarm=62)
 *   ダイナミックレンジ: ≈66 dB
 *   ISR CPU 占有: < 0.1%     (FPS への影響ゼロ)
 */

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
 
 static const char * const TAG = "ADPCM";
 
 /* ============================================================
  * LEDC PWM 設定
  *
  * LEDC_TIMER_11_BIT = 2048段階
  * キャリア = 80MHz / 2048 = 39,062 Hz (可聴域外)
  * ============================================================ */
 #define PWM_RESOLUTION    LEDC_TIMER_11_BIT   /* 11bit = 0〜2047 */
 #define PWM_MAX_DUTY      2047U               /* フルスケール */
 #define PWM_MID_DUTY      1024U               /* 無音 (中点) */
 #define PWM_FREQ_HZ       39062U              /* 80MHz/2048 */
 #define PWM_CHANNEL       LEDC_CHANNEL_0
 #define PWM_TIMER         LEDC_TIMER_0
 #define PWM_SPEED_MODE    LEDC_LOW_SPEED_MODE
 
 /* ============================================================
  * gptimer (サンプルクロック) 設定
  *
  * resolution_hz = 1MHz (80MHzの整数分周、誤差ゼロ)
  * alarm_count   = 62   → 1MHz/62 = 16,129 Hz ≈ 16kHz (誤差+0.8%)
  *
  * alarm=62 を選択した理由:
  *   alarm=63 → 15,873 Hz (誤差-0.8%)
  *   alarm=62 → 16,129 Hz (誤差+0.8%)
  *   いずれも ±1% 以内。フレーム同期はサンプル数カウントで
  *   行うため誤差は蓄積しない。
  *   16,129 Hz の方が 16,000 Hz に近い (算術的には 62.5 が最適)。
  * ============================================================ */
 #define TIMER_RESOLUTION_HZ  1000000U   /* 1MHz */
 #define TIMER_ALARM_COUNT    62U        /* → 16,129 Hz ≈ 16kHz */
 #define TIMER_ACTUAL_HZ      (TIMER_RESOLUTION_HZ / TIMER_ALARM_COUNT)
 
 /* リングバッファ低水位 */
 #define RING_LOW_WATER  1024U
 
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
 static const int8_t s_idx_table[8] = { -1,-1,-1,-1, 2, 4, 6, 8 };
 
 /* ============================================================
  * リングバッファ (int16 PCM)
  * ============================================================ */
 static volatile int16_t  s_ring[ADPCM_RING_SIZE];
 static volatile uint32_t s_ring_wr;
 static volatile uint32_t s_ring_rd;
 
 static inline uint32_t ring_used(void)
 {
     return (s_ring_wr - s_ring_rd) & (ADPCM_RING_SIZE - 1U);
 }
 static inline int ring_full(void)  { return ring_used() >= (ADPCM_RING_SIZE - 1U); }
 static inline int ring_empty(void) { return s_ring_wr == s_ring_rd; }
 
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
 
 static volatile uint32_t s_isr_samples;
 static volatile uint32_t s_frames_given;
 
 /* ============================================================
  * WAV ヘッダパーサ
  * ============================================================ */
 static int wav_parse(void)
 {
     const uint8_t *d = s_wav_data;
     if (s_wav_size < 44U) { ESP_LOGE(TAG, "WAV too small"); return -1; }
     if (d[0]!='R'||d[1]!='I'||d[2]!='F'||d[3]!='F') { ESP_LOGE(TAG, "No RIFF"); return -1; }
     if (d[8]!='W'||d[9]!='A'||d[10]!='V'||d[11]!='E') { ESP_LOGE(TAG, "No WAVE"); return -1; }
 
     uint32_t pos = 12U;
     uint32_t af = 0, sr = 0, ba = 0;
     int got_fmt = 0, got_data = 0;
 
     while ((pos + 8U) <= s_wav_size) {
         uint32_t csz = (uint32_t)d[pos+4] | ((uint32_t)d[pos+5]<<8)
                      | ((uint32_t)d[pos+6]<<16) | ((uint32_t)d[pos+7]<<24);
         if (d[pos]=='f'&&d[pos+1]=='m'&&d[pos+2]=='t'&&d[pos+3]==' ') {
             af = (uint32_t)d[pos+8] | ((uint32_t)d[pos+9]<<8);
             sr = (uint32_t)d[pos+12]|((uint32_t)d[pos+13]<<8)
                | ((uint32_t)d[pos+14]<<16)|((uint32_t)d[pos+15]<<24);
             ba = (uint32_t)d[pos+20]|((uint32_t)d[pos+21]<<8);
             got_fmt = 1;
             ESP_LOGI(TAG, "fmt: af=0x%04"PRIu32" sr=%"PRIu32" ba=%"PRIu32, af, sr, ba);
         } else if (d[pos]=='d'&&d[pos+1]=='a'&&d[pos+2]=='t'&&d[pos+3]=='a') {
             s_data_offset = pos + 8U;
             s_data_size   = csz;
             got_data = 1;
         }
         pos += 8U + csz;
         if (csz & 1U) pos++;
         if (got_fmt && got_data) break;
     }
     if (!got_fmt || !got_data) { ESP_LOGE(TAG, "chunk not found"); return -1; }
     if (af != (uint32_t)WAV_FMT_IMA_ADPCM) { ESP_LOGE(TAG, "Not IMA ADPCM"); return -1; }
 
     s_block_align = ba;
     s_sample_rate = sr;
     s_read_pos    = s_data_offset;
     ESP_LOGI(TAG, "WAV OK sr=%"PRIu32" ba=%"PRIu32" sz=%"PRIu32, sr, ba, s_data_size);
     return 0;
 }
 
 /* ============================================================
  * IMA ADPCM ニブルデコーダ
  * ============================================================ */
 static IRAM_ATTR int16_t adpcm_nibble(adpcm_state_t *st, uint8_t n)
 {
     int16_t step = s_step_table[st->step_index];
     int32_t diff = (int32_t)step >> 3;
     if (n & 4U) diff += step;
     if (n & 2U) diff += step >> 1;
     if (n & 1U) diff += step >> 2;
     if (n & 8U) diff = -diff;
     int32_t pred = (int32_t)st->predictor + diff;
     if      (pred >  32767) pred =  32767;
     else if (pred < -32768) pred = -32768;
     st->predictor = (int16_t)pred;
     int8_t idx = st->step_index + s_idx_table[n & 7U];
     if      (idx <  0) idx =  0;
     else if (idx > 88) idx = 88;
     st->step_index = idx;
     return st->predictor;
 }
 
 /* ============================================================
  * リングバッファへのデコード (adpcm_task から呼ぶ)
  * ============================================================ */
 static void decode_block_to_ring(const uint8_t *blk, uint32_t bytes)
 {
     if (bytes < 4U) return;
 
     adpcm_state_t st;
     st.predictor  = (int16_t)((uint16_t)blk[0] | ((uint16_t)blk[1]<<8));
     st.step_index = (int8_t)blk[2];
     if (st.step_index <  0) st.step_index =  0;
     if (st.step_index > 88) st.step_index = 88;
 
     /* ボリューム適用してリングへ書く */
 #define RING_PUSH(raw) do {                                                 \
     int16_t _s = (int16_t)(((int32_t)(raw) * (int32_t)CFG_AUDIO_VOL)>>8); \
     while (ring_full()) {                                                   \
         if (ring_used() < RING_LOW_WATER) { taskYIELD(); }                 \
         else                              { vTaskDelay(1); }                \
     }                                                                       \
     s_ring[s_ring_wr & (ADPCM_RING_SIZE-1U)] = _s;                         \
     s_ring_wr++;                                                            \
 } while(0)
 
     RING_PUSH(st.predictor);
     for (uint32_t i = 4U; i < bytes; i++) {
         RING_PUSH(adpcm_nibble(&st,  blk[i]       & 0x0FU));
         RING_PUSH(adpcm_nibble(&st, (blk[i] >> 4) & 0x0FU));
     }
 #undef RING_PUSH
 }
 
 /* ============================================================
  * gptimer ISR — 約16kHz で発火
  *
  * 処理:
  *   1. ring から int16 PCM サンプルを1個取得
  *   2. [-32768, +32767] → [0, PWM_MAX_DUTY] に線形変換
  *   3. ledc_set_duty() + ledc_update_duty() で PWM duty 更新
  *   4. フレーム同期セマフォ give
  *
  * PCM → PWM duty 変換:
  *   PCM は signed 16bit。中点 (0) を PWM_MID_DUTY(1024) にマップ。
  *   duty = PCM/32768 × 1024 + 1024
  *        = (PCM >> 5) + 1024
  *   ただし CFG_AUDIO_VOL で既にスケール済みなので
  *   オーバーフロー対策のクランプを入れる。
  *
  * ISR 予算:
  *   発火間隔 = 160MHz / 16129Hz ≈ 9920 cycles
  *   ledc_set_duty():  ~40 cycles (レジスタ直接書き込み)
  *   ledc_update_duty():~30 cycles
  *   合計: ~100 cycles → 占有率 ~1% → FPS 影響ゼロ
  * ============================================================ */
 static IRAM_ATTR bool audio_isr_cb(gptimer_handle_t timer,
                                     const gptimer_alarm_event_data_t *edata,
                                     void *user_ctx)
 {
     (void)timer; (void)edata; (void)user_ctx;
     BaseType_t woken = pdFALSE;
 
     /* サンプル取得 */
     int32_t pcm;
     if (!ring_empty()) {
         pcm = (int32_t)s_ring[s_ring_rd & (ADPCM_RING_SIZE-1U)];
         s_ring_rd++;
     } else {
         pcm = 0;  /* アンダーラン: 無音 (中点) */
     }
 
     /* PCM [-32768,+32767] → duty [0, 2047]
      * duty = clamp(pcm >> 5, -1024, 1023) + 1024
      * pcm >> 5 は算術右シフト (int32 は符号拡張) */
     int32_t duty = (pcm >> 5) + (int32_t)PWM_MID_DUTY;
     if      (duty > (int32_t)PWM_MAX_DUTY) duty = (int32_t)PWM_MAX_DUTY;
     else if (duty < 0)                     duty = 0;
 
     /* LEDC duty 更新 (ISR safe: レジスタ直接書き込み) */
     ledc_set_duty(PWM_SPEED_MODE, PWM_CHANNEL, (uint32_t)duty);
     ledc_update_duty(PWM_SPEED_MODE, PWM_CHANNEL);
 
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
 
     return (woken == pdTRUE);
 }
 
 /* ============================================================
  * オーディオデコードタスク
  * ============================================================ */
 static void adpcm_task(void *arg)
 {
     (void)arg;
     ESP_LOGI(TAG, "adpcm_task start core=%d pri=%d",
              xPortGetCoreID(), uxTaskPriorityGet(NULL));
     for (;;) {
         if (s_read_pos >= (s_data_offset + s_data_size)) {
             vTaskDelay(pdMS_TO_TICKS(50));
             continue;
         }
         if (ring_used() > RING_LOW_WATER && ring_full()) {
             vTaskDelay(1);
             continue;
         }
         uint32_t rem = (s_data_offset + s_data_size) - s_read_pos;
         uint32_t blk = (s_block_align > 0U) ? s_block_align : 512U;
         if (blk > rem) blk = rem;
         decode_block_to_ring(s_wav_data + s_read_pos, blk);
         s_read_pos += blk;
         if (ring_used() > RING_LOW_WATER) vTaskDelay(1);
     }
 }
 
 /* ============================================================
  * ring 事前充填 (init / rewind 共通)
  * ============================================================ */
 static void prefill_ring(void)
 {
     uint32_t filled = 0U;
     while (filled < (ADPCM_RING_SIZE - 1U)) {
         if (s_read_pos >= (s_data_offset + s_data_size)) break;
         uint32_t rem = (s_data_offset + s_data_size) - s_read_pos;
         uint32_t blk = (s_block_align > 0U) ? s_block_align : 512U;
         if (blk > rem) blk = rem;
 
         const uint8_t *p = s_wav_data + s_read_pos;
         adpcm_state_t st;
         st.predictor  = (int16_t)((uint16_t)p[0] | ((uint16_t)p[1]<<8));
         st.step_index = (int8_t)p[2];
         if (st.step_index < 0)  st.step_index = 0;
         if (st.step_index > 88) st.step_index = 88;
 
         {
             /* ヘッダサンプル */
             if (!ring_full()) {
                 int16_t s = (int16_t)(((int32_t)st.predictor
                                        * (int32_t)CFG_AUDIO_VOL) >> 8);
                 s_ring[s_ring_wr & (ADPCM_RING_SIZE-1U)] = s;
                 s_ring_wr++; filled++;
             }
             for (uint32_t i = 4U; i < blk && !ring_full(); i++) {
                 int16_t s0 = adpcm_nibble(&st,  p[i]       & 0x0FU);
                 int16_t s1 = adpcm_nibble(&st, (p[i] >> 4) & 0x0FU);
                 int16_t v0 = (int16_t)(((int32_t)s0*(int32_t)CFG_AUDIO_VOL)>>8);
                 int16_t v1 = (int16_t)(((int32_t)s1*(int32_t)CFG_AUDIO_VOL)>>8);
                 s_ring[s_ring_wr & (ADPCM_RING_SIZE-1U)] = v0; s_ring_wr++; filled++;
                 if (ring_full()) break;
                 s_ring[s_ring_wr & (ADPCM_RING_SIZE-1U)] = v1; s_ring_wr++; filled++;
             }
         }
         s_read_pos += blk;
     }
     ESP_LOGI(TAG, "prefill: %"PRIu32" samples (%.1fs)",
              filled, (float)filled / (float)s_sample_rate);
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
     s_isr_samples  = 0U;
     s_frames_given = 0U;
 
     if (wav_parse() != 0) return -1;
 
     /* ---- LEDC PWM 初期化 ------------------------------------
      * タイマ: 11bit 分解能、39kHz キャリア、LOW_SPEED モード
      * チャンネル: CFG_AUDIO_PIN_P に出力、初期 duty = 中点
      * --------------------------------------------------------- */
     ledc_timer_config_t lt = {
         .speed_mode      = PWM_SPEED_MODE,
         .duty_resolution = PWM_RESOLUTION,
         .timer_num       = PWM_TIMER,
         .freq_hz         = PWM_FREQ_HZ,
         .clk_cfg         = LEDC_AUTO_CLK,
     };
     ESP_ERROR_CHECK(ledc_timer_config(&lt));
 
     ledc_channel_config_t lc = {
         .gpio_num   = CFG_AUDIO_PIN_P,
         .speed_mode = PWM_SPEED_MODE,
         .channel    = PWM_CHANNEL,
         .timer_sel  = PWM_TIMER,
         .duty       = PWM_MID_DUTY,   /* 無音 = 中点 */
         .hpoint     = 0,
         .intr_type  = LEDC_INTR_DISABLE,
     };
     ESP_ERROR_CHECK(ledc_channel_config(&lc));
 
     ESP_LOGI(TAG, "LEDC PWM: GPIO%d  carrier=%uHz  resolution=11bit  duty_mid=%u",
              CFG_AUDIO_PIN_P, PWM_FREQ_HZ, PWM_MID_DUTY);
 
     /* ---- gptimer (サンプルクロック) 初期化 ------------------
      * 1MHz タイマ、alarm=62 → 16,129Hz ≈ 16kHz
      * --------------------------------------------------------- */
     gptimer_config_t tcfg;
     memset(&tcfg, 0, sizeof(tcfg));
     tcfg.clk_src       = GPTIMER_CLK_SRC_DEFAULT;
     tcfg.direction     = GPTIMER_COUNT_UP;
     tcfg.resolution_hz = TIMER_RESOLUTION_HZ;   /* 1MHz */
 
     esp_err_t err = gptimer_new_timer(&tcfg, &s_gptimer);
     if (err != ESP_OK) {
         ESP_LOGE(TAG, "gptimer_new_timer: 0x%x", err);
         return -1;
     }
 
     gptimer_alarm_config_t acfg;
     memset(&acfg, 0, sizeof(acfg));
     acfg.alarm_count                = TIMER_ALARM_COUNT;   /* 62 */
     acfg.reload_count               = 0U;
     acfg.flags.auto_reload_on_alarm = 1U;
     ESP_ERROR_CHECK(gptimer_set_alarm_action(s_gptimer, &acfg));
 
     gptimer_event_callbacks_t cbs;
     memset(&cbs, 0, sizeof(cbs));
     cbs.on_alarm = audio_isr_cb;
     ESP_ERROR_CHECK(gptimer_register_event_callbacks(s_gptimer, &cbs, NULL));
     ESP_ERROR_CHECK(gptimer_enable(s_gptimer));
 
     ESP_LOGI(TAG, "gptimer: 1MHz / %u = %u Hz (sample clock)",
              TIMER_ALARM_COUNT, TIMER_ACTUAL_HZ);
 
     /* ring 事前充填 → タイマ開始 */
     ESP_LOGI(TAG, "pre-filling ring (%u samples)...", ADPCM_RING_SIZE);
     prefill_ring();
     ESP_ERROR_CHECK(gptimer_start(s_gptimer));
 
     xTaskCreatePinnedToCore(adpcm_task, "adpcm", 4096U, NULL,
                             4, NULL, 0);
 
     ESP_LOGI(TAG, "adpcm_init OK  PWM=%uHz  SR≈%uHz  SPF=%"PRIu32,
              PWM_FREQ_HZ, TIMER_ACTUAL_HZ,
              (s_sample_rate * (uint32_t)CFG_FRAME_MS) / 1000U);
     return 0;
 }
 
 /* ============================================================
  * adpcm_rewind — public API
  * ============================================================ */
 void adpcm_rewind(void)
 {
     ESP_ERROR_CHECK(gptimer_stop(s_gptimer));
 
     /* PWM を中点 (無音) に戻す */
     ledc_set_duty(PWM_SPEED_MODE, PWM_CHANNEL, PWM_MID_DUTY);
     ledc_update_duty(PWM_SPEED_MODE, PWM_CHANNEL);
 
     s_read_pos     = s_data_offset;
     s_ring_wr      = 0U;
     s_ring_rd      = 0U;
     s_isr_samples  = 0U;
     s_frames_given = 0U;
 
     ESP_LOGI(TAG, "rewind: pre-filling ring...");
     prefill_ring();
 
     ESP_ERROR_CHECK(gptimer_start(s_gptimer));
     ESP_LOGI(TAG, "rewind done");
 }
 
