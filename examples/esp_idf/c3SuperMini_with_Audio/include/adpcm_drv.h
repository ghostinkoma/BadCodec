/**
 * @file  adpcm_drv.h
 * @brief IMA ADPCM decoder + LEDC PWM / Hardware SDM audio output
 * @version v1.0.0
 *
 * Signal chain (mode=SINGLE/BTL):
 *   Flash WAV (IMA-ADPCM 4-bit) -> PCM ring buffer
 *   -> gptimer ISR ~16kHz -> LEDC duty update -> GPIO PWM
 *
 * Signal chain (mode=SDM):
 *   Flash WAV (IMA-ADPCM 4-bit) -> PCM ring buffer
 *   -> gptimer ISR ~SR Hz -> sdm_channel_set_duty() -> HW SDM -> GPIO
 */

 #ifndef ADPCM_DRV_H
 #define ADPCM_DRV_H
 
 #include <stdint.h>
 #include "freertos/FreeRTOS.h"
 #include "freertos/semphr.h"
 /* config.h はここでインクルードしない。
  * adpcm_drv.h → config.h → adpcm_drv.h の循環 include を防ぐため。
  * CFG_* は adpcm_drv.c が config.h を先頭で include して使用する。 */

 /* WAV fmt chunk offsets */
 #define WAV_FMT_AUDIO_FORMAT_OFF    8U
 #define WAV_FMT_SAMPLE_RATE_OFF    12U
 #define WAV_FMT_BLOCK_ALIGN_OFF    20U
 #define WAV_FMT_IMA_ADPCM       0x0011U
 
 /* Ring buffer: 65536 samples = 4.1sec @ 16kHz */
 #define ADPCM_RING_SIZE  65536U
 
 /* Public API */
 int  adpcm_init(const uint8_t *adpcm_data, uint32_t adpcm_size,
                 SemaphoreHandle_t video_sem);
 void adpcm_rewind(void);
 
 /** フレーム間隔設定 ms 単位 (後方互換) */
 void adpcm_set_frame_ms(uint32_t ms);

/** フレーム間隔設定 μs 単位 (推奨: 29.97fps=33366μs, 誤差0.003%) */
 void adpcm_set_frame_us(uint32_t us);

/** 音量を動的に変更する (0=無音, 256=フルスケール)
 *  ボタン VOL+/- 押下時に button_get_vol() の値を渡す。
 *  ISR から参照する s_vol を更新するため即時反映される。 */
 void adpcm_set_vol(uint16_t vol);

/** 音声を一時停止 (gptimer 停止 + 無音出力) */
 void adpcm_pause(void);

/** 音声を再開 (gptimer 再起動、カウンタリセット) */
 void adpcm_resume(void);
 
 /* VU レベル取得 (draw モジュールから参照)
  * 戻り値: 0〜32767 (直近サンプルの絶対値ピーク)         */
 uint16_t adpcm_get_vu(void);
 
 /* 実際のサンプルレートを返す (WAV ヘッダ解析後に有効)   */
 uint32_t adpcm_get_sample_rate(void);
 
 /* ADPCM デコーダ内部状態 (adpcm_drv.c 内部使用)        */
 typedef struct {
     int16_t predictor;
     int8_t  step_index;
 } adpcm_state_t;
 
 #endif /* ADPCM_DRV_H */
 
