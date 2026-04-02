/**
 * @file  config.h
 * @brief ESP32-C3 Super Mini + SSD1306 128x64 + ADPCM audio
 * @version v0.8.1
 */

 #ifndef CONFIG_H
 #define CONFIG_H
 
 /* ---- I2C (OLED) ------------------------------------------ */
 #define CFG_I2C_PORT        I2C_NUM_0
 #define CFG_I2C_SDA         8
 #define CFG_I2C_SCL         9
 #define CFG_I2C_FREQ_HZ     1000000
 
 /* ---- SSD1306 128x64 -------------------------------------- */
 #define CFG_OLED_ADDR       0x3D
 #define CFG_PHYS_W          128
 #define CFG_PHYS_H          64
 #define CFG_PAGES           8
 #define CFG_DISP_W          128
 #define CFG_DISP_H          64
 #define CFG_X_OFFSET        0
 #define CFG_Y_OFFSET        0
 
 /* ---- BadCodec video -------------------------------------- */
 #define CFG_VIDEO_W         128
 #define CFG_VIDEO_H         64
 
 /* ---- Audio: differential sigma-delta PDM output ----------
  * Two GPIO pins required for balanced (differential) output.
  *   CFG_AUDIO_PIN_P : positive (non-inverting) Sigma-Delta bit
  *   CFG_AUDIO_PIN_N : negative (inverting)     Sigma-Delta bit
  * Connect each through 100ohm + 10nF RC to a differential
  * amplifier (e.g. PAM8302 in BTL mode) or balanced speaker.
  * Avoid GPIO11-17 (internal flash).
  * GPIO8/9 with NO external pull-ups: OK (used for I2C above). */
 #define CFG_AUDIO_PIN_P     10      /* positive phase GPIO */
 #define CFG_AUDIO_PIN_N     3       /* negative phase GPIO */
 
 /* Audio sampling rate — adpcm4.h WAVヘッダ実測値: 16000Hz
  * ffmpeg: ffmpeg -i input.mp4 -ac 1 -ar 16000 -c:a adpcm_ima_wav out.wav */
 #define CFG_AUDIO_SR        173000U

 /* ΣΔ変調オーバーサンプリング比は adpcm_drv.c の SDM_OSR マクロで設定。
  * デフォルト: SDM_OSR=32 → ISR @ 512kHz → 音質向上 (低音再現)
  * LPF推奨値: R=1kΩ, C=10nF (fc≈16kHz), カップリングC=10μF (アンプ経由) */

 /* Volume: 0 = mute, 256 = full scale (may clip) */
 #define CFG_AUDIO_VOL       200U
 
 /* ---- Frame timing ----------------------------------------
  * Nominal frame interval in milliseconds (~34.5 fps).
  * When audio is running the frame rate is derived from the
  * audio sample clock; CFG_FRAME_MS is a fallback only.       */
 #define CFG_FRAME_MS        32U
 
 /* ---- Audio/Video sync (参考値) ---------------------------
  * 実際の計算は adpcm_drv.c の ISR 内で s_sample_rate から行う。
  * WAVヘッダのSRが config.h と異なっても自動的に正しく動作する。
  *   例: SR=16000, frame_ms=29 → 16000*29/1000 = 464 samples/frame
  * ---------------------------------------------------------- */
 #define ADPCM_SAMPLES_PER_FRAME \
     ( (CFG_AUDIO_SR) * (CFG_FRAME_MS) / 1000U )
 
 /* ---- Source ---------------------------------------------- */
 #define CFG_SOURCE_FLASH    1
 /* #define CFG_SOURCE_SD   1 */
 #ifdef CFG_SOURCE_SD
 #define CFG_SD_FILE         "/sdcard/output.bad"
 #define CFG_AUDIO_SD_FILE   "/sdcard/adpcm4.wav"
 #endif

 /* ---- Debug: Video / Audio enable -------------------------
  *
  *   CFG_VIDEO_ENABLE   1 = 映像再生あり  / 0 = 映像スキップ（OLEDは初期化のみ）
  *   CFG_AUDIO_ENABLE   1 = 音声再生あり  / 0 = 音声なし（セマフォはソフトタイマで代替）
  *
  *   切り分け手順:
  *     1) VIDEO=1, AUDIO=0 → 映像のみ再生できるか確認
  *     2) VIDEO=0, AUDIO=1 → 音声のみ（シリアルログで ADPCM_SAMPLES 確認）
  *     3) VIDEO=1, AUDIO=1 → 両方同時（本来の動作）
  * ---------------------------------------------------------- */
 #define CFG_VIDEO_ENABLE    1   /* 1=ON  0=OFF */
 #define CFG_AUDIO_ENABLE    1   /* 1=ON  0=OFF */

 #endif /* CONFIG_H */
 
