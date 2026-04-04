/**
 * @file  config.h
 * @brief ESP32-C3 Super Mini + SSD1306 128x64 + LEDC PWM audio
 * @version v0.9.0
 *
 * v0.9.0 変更点:
 *   音声出力方式を ソフトウェアΣΔ PDM → LEDC ハードウェア PWM に刷新。
 *   CFG_AUDIO_DIFFERENTIAL 廃止。
 *   LPF 定数を PWM キャリア (39kHz) 除去用に変更。
 *
 * LPF 必須変更:
 *   旧 (ΣΔ用): R=1kΩ,  C=10nF  (fc≈16kHz) ← PWM では不十分
 *   新 (PWM用): R=10kΩ, C=100nF (fc≈160Hz) ← キャリアを -48dB 減衰
 *
 * 接続:
 *   GPIO10 → R=10kΩ → ┬── C=100nF ── GND  (LPF)
 *                      └── C=10μF  ── アンプ入力 (カップリング)
 */

 #ifndef CONFIG_H
 #define CONFIG_H
 
 /* ---- I2C (OLED) ------------------------------------------ */
 #define CFG_I2C_PORT        I2C_NUM_0
 #define CFG_I2C_SDA         8
 #define CFG_I2C_SCL         9
 #define CFG_I2C_FREQ_HZ     400000
 
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
 
 /* ---- Audio: LEDC PWM output ------------------------------
  *
  * v2.0.0: ソフトウェアΣΔを廃止し LEDC ハードウェア PWM を使用。
  *
  * PWM キャリア: 39,062 Hz (80MHz / 2^11)  ← 可聴域 (20kHz) を超える
  * PWM 分解能:   11 bit (0〜2047 = 2048段階)
  * ダイナミックレンジ: 20×log10(2048) ≈ 66 dB
  *
  * CFG_AUDIO_PIN_P: PWM 出力 GPIO (LEDC CHANNEL_0)
  *
  * 接続 (必須):
  *   GPIO10 → R=10kΩ → ┬── C=100nF ── GND   (LPF fc≈160Hz)
  *                      └── C=10μF  ── アンプ入力 (カップリング)
  *
  *   ※ ΣΔ時代の R=1kΩ, C=10nF では fc=16kHz でキャリアが通過する。
  *     R=10kΩ, C=100nF (fc=160Hz) に必ず変更すること。
  *
  * CFG_AUDIO_DIFFERENTIAL は廃止。差動出力は不要。
  * ---------------------------------------------------------- */
 #define CFG_AUDIO_PIN_P     10      /* PWM 出力 GPIO */
 
 /* Audio sampling rate */
 #define CFG_AUDIO_SR        16000U
 
 /* Volume: 0=mute, 256=full scale */
 #define CFG_AUDIO_VOL       120U
 
 /* ---- Frame timing ---------------------------------------- */
 #define CFG_FRAME_MS        29U
 
 /* ---- A/V 同期 (参考値) ----------------------------------- */
 #define ADPCM_SAMPLES_PER_FRAME \
     ( (CFG_AUDIO_SR) * (CFG_FRAME_MS) / 1000U )
 
 /* ---- Source ---------------------------------------------- */
 #define CFG_SOURCE_FLASH    1
 /* #define CFG_SOURCE_SD   1 */
 #ifdef CFG_SOURCE_SD
 #define CFG_SD_FILE         "/sdcard/output.bad"
 #define CFG_AUDIO_SD_FILE   "/sdcard/adpcm4.wav"
 #endif
 
 /* ---- Debug ----------------------------------------------- */
 #define CFG_VIDEO_ENABLE    1
 #define CFG_AUDIO_ENABLE    1
 
 #endif /* CONFIG_H */
 
