/**
 * @file  adpcm_drv.h
 * @brief IMA ADPCM 4-bit decoder + differential sigma-delta PDM output
 * @version v0.8.1
 *
 * Signal chain:
 *   Flash WAV (IMA-ADPCM 4-bit, 18 kHz mono)
 *     -> adpcm_drv decode task  -> int16 PCM ring buffer
 *     -> gptimer ISR @ 1.152MHz -> 1st-order sigma-delta modulator
 *     -> CFG_AUDIO_PIN_P  positive phase
 *     -> CFG_AUDIO_PIN_N  negative phase (complementary)
 *     -> RC low-pass + differential amp -> speaker
 */

 #ifndef ADPCM_DRV_H
 #define ADPCM_DRV_H
 
 #include <stdint.h>
 #include "freertos/FreeRTOS.h"
 #include "freertos/semphr.h"
 #include "config.h"
 
 /* ============================================================
  * WAV chunk field types (corrected from previous version)
  *
  * All WAV header fields are read as byte arrays and assembled
  * with explicit shifts to avoid alignment and endian issues.
  * The parsed values are stored as uint32_t throughout.
  * (Previous version incorrectly used uint64_t for sample rate.)
  * ============================================================ */
 
 /* fmt sub-chunk offsets (relative to chunk ID byte, pos):
  *   pos+ 0   4B  'fmt '  chunk ID
  *   pos+ 4   4B  chunk data size   (uint32 LE)
  *   pos+ 8   2B  AudioFormat       (uint16 LE)  0x0011 = IMA ADPCM
  *   pos+10   2B  NumChannels       (uint16 LE)
  *   pos+12   4B  SampleRate        (uint32 LE)  <- read here
  *   pos+16   4B  ByteRate          (uint32 LE)
  *   pos+20   2B  BlockAlign        (uint16 LE)  <- read here
  *   pos+22   2B  BitsPerSample     (uint16 LE)
  *
  * NOTE: previous version read SampleRate from pos+24 and
  *       BlockAlign from pos+32 — both were WRONG.
  */
 #define WAV_FMT_AUDIO_FORMAT_OFF   8U   /* uint16 at pos+8  */
 #define WAV_FMT_SAMPLE_RATE_OFF   12U   /* uint32 at pos+12 */
 #define WAV_FMT_BLOCK_ALIGN_OFF   20U   /* uint16 at pos+20 */
 
 /* Expected AudioFormat value for IMA ADPCM */
 #define WAV_FMT_IMA_ADPCM         0x0011U
 
 /* ============================================================
  * Ring buffer size (must be power-of-2)
  *
  * 256KB(131072) を試みたが DRAM が 4352 bytes オーバーしたため
  * 65536 samples (128KB, 4.1秒分) に設定する。
  *
  *   バッファ時間: 65536 / 16000Hz = 4.1 秒 = 141 フレーム分
  *   → adpcm_task が動けない間 (I2C flush ~23ms) でも
  *     ring が枯渇しない余裕が十分確保される
  *   → ring underrun によるグリッチ・ノイズが激減
  *
  *   メモリコスト: 65536 × 2 bytes = 128KB
  * ============================================================ */
 #define ADPCM_RING_SIZE   65536U
 
 /* ============================================================
  * Public API
  * ============================================================ */
 
 /**
  * adpcm_init
  *
  * Parse WAV header, configure GPIO pins for differential PDM,
  * start gptimer ISR at CFG_AUDIO_SR * CFG_AUDIO_OSR Hz,
  * and launch the audio decode FreeRTOS task on core 1.
  *
  * @param adpcm_data  WAV file bytes in Flash.  Pass NULL to use
  *                    the built-in data from adpcm4.h.
  * @param adpcm_size  Total byte count (ignored when NULL).
  * @param video_sem   Binary semaphore.  The ISR calls
  *                    xSemaphoreGiveFromISR() once every
  *                    ADPCM_SAMPLES_PER_FRAME audio samples,
  *                    unblocking the video task for one frame.
  * @return  0 = OK,  -1 = error (check serial log).
  */
 int  adpcm_init(const uint8_t    *adpcm_data,
                 uint32_t          adpcm_size,
                 SemaphoreHandle_t video_sem);
 
 /** Reset audio playback to the beginning (call on video loop). */
 void adpcm_rewind(void);
 
 /* ============================================================
  * Internal decoder state  (used only inside adpcm_drv.c)
  * ============================================================ */
 typedef struct {
     int16_t predictor;    /* reconstructed sample         */
     int8_t  step_index;   /* index into step_table[0..88] */
 } adpcm_state_t;
 
 #endif /* ADPCM_DRV_H */
 
