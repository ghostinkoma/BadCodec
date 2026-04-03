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
 
 /* ---- Audio: sigma-delta PDM output -----------------------
  *
  * CFG_AUDIO_DIFFERENTIAL  1 = 差動出力 (PIN_P + PIN_N 両方使用)
  *                          0 = シングルエンド出力 (PIN_P のみ)
  *
  * ノイズ切り分け用:
  *   差動出力では ISR 内で 2回の REG_WRITE が必要。
  *   2回の書き込みの間 (~3ns) に両ピンが同レベルになるグリッチが
  *   160kHz で発生し LPF 後のノイズ源になる可能性がある。
  *   また GPIO3 (PIN_N) は ESP32-C3 の UART0_RX と共用されることがあり
  *   USB シリアル通信と干渉する可能性がある。
  *
  *   まず CFG_AUDIO_DIFFERENTIAL=0 で切り分けることを推奨。
  *
  * シングルエンド接続 (CFG_AUDIO_DIFFERENTIAL=0):
  *   GPIO10 → R=1kΩ → ┬─ C=10nF  → GND       (LPF fc≈16kHz)
  *                     └─ C=10μF  → アンプ入力 (カップリング)
  *
  * 差動接続 (CFG_AUDIO_DIFFERENTIAL=1):
  *   GPIO10 → R=100Ω → 差動アンプ +入力
  *   GPIO3  → R=100Ω → 差動アンプ −入力
  *   → 電源ノイズをCMRRでキャンセルできる (理論上)
  *   → ただし2ピン間のスキューがノイズ源になることもある
  *
  * CFG_AUDIO_PIN_P : 正相 GPIO (シングルエンド時もこちらを使用)
  * CFG_AUDIO_PIN_N : 逆相 GPIO (差動時のみ使用, 0=sigle では固定LOW)
  * ---------------------------------------------------------- */
 #define CFG_AUDIO_DIFFERENTIAL  1       /* 0=シングルエンド(推奨)  1=差動 */
 #define CFG_AUDIO_PIN_P         10      /* positive phase GPIO */
 #define CFG_AUDIO_PIN_N         3       /* negative phase GPIO (差動時のみ) */
 
 /* Audio sampling rate — adpcm4.h WAVヘッダ実測値: 16000Hz
  * ffmpeg: ffmpeg -i input.mp4 -ac 1 -ar 16000 -c:a adpcm_ima_wav out.wav */
 #define CFG_AUDIO_SR        16000U

 /* ΣΔ変調オーバーサンプリング比は adpcm_drv.c の SDM_OSR マクロで設定。
  * デフォルト: SDM_OSR=32 → ISR @ 512kHz → 音質向上 (低音再現)
  * LPF推奨値: R=1kΩ, C=10nF (fc≈16kHz), カップリングC=10μF (アンプ経由) */

 /* Volume: 0 = mute, 256 = full scale (may clip) */
 #define CFG_AUDIO_VOL       200U
 
 /* ---- Frame timing ----------------------------------------
  * Nominal frame interval in milliseconds (~34.5 fps).
  * When audio is running the frame rate is derived from the
  * audio sample clock; CFG_FRAME_MS is a fallback only.       */
 #define CFG_FRAME_MS        29U
 
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
 
