/**
 * @file  adpcm_drv.h
 * @brief IMA ADPCM decoder + audio output driver (public API)
 * @version v3.0.0
 *
 * !! config.h を include しない !!
 *    adpcm_drv.h → config.h → adpcm_drv.h の循環 include を防ぐため
 *    config.h は adpcm_drv.c のみが include する。
 *
 * Signal chain (CFG_AUDIO_OUTPUT_MODE=0 SINGLE / 1 BTL):
 *   Flash WAV → PCM ring → gptimer ISR → ledc_set_duty() → GPIO PWM
 *
 * Signal chain (CFG_AUDIO_OUTPUT_MODE=2 SDM):
 *   Flash WAV → PCM ring → gptimer ISR → TaskNotify → sdm_task →
 *   sdm_channel_set_duty() → HW SDM → GPIO
 *   (ISR から Flash 関数を呼べないため TaskNotify 経由)
 */

 #ifndef ADPCM_DRV_H
 #define ADPCM_DRV_H
 
 #include <stdint.h>
 #include "freertos/FreeRTOS.h"
 #include "freertos/semphr.h"
 /* config.h は include しない */
 
 /* リングバッファサイズ: 65536 samples = 4.1 sec @ 16kHz */
 #define ADPCM_RING_SIZE  65536U
 
 /* ADPCM デコーダ内部状態 */
 typedef struct {
     int16_t predictor;
     int8_t  step_index;
 } adpcm_state_t;
 
 /* ---- Public API ------------------------------------------ */
 
 /**
  * 初期化。adpcm_data=NULL のとき adpcm4.h の bad_audio_data を使用。
  * 成功=0, 失敗=-1
  */
 int adpcm_init(const uint8_t *adpcm_data, uint32_t adpcm_size,
                SemaphoreHandle_t video_sem);
 
 /** 先頭に巻き戻して再生を再開する (ループ用) */
 void adpcm_rewind(void);
 
 /** フレーム間隔 [ms] を設定する。bad_init() 後に main.c から呼ぶ。 */
 void adpcm_set_frame_ms(uint32_t ms);
 
 /** VU ピーク値を返す [0-32767]。呼ぶたびに減衰する。 */
 uint16_t adpcm_get_vu(void);
 
 /** WAV ヘッダから取得したサンプルレートを返す */
 uint32_t adpcm_get_sample_rate(void);
 
 #endif /* ADPCM_DRV_H */
 
