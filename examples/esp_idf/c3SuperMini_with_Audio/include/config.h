/**
 * @file  config.h
 * @brief ESP32-C3 Super Mini + SSD1306 128x64 + LEDC/SDM audio
 * @version v1.0.0
 *
 * 変更点 (v1.0.0):
 *   - CFG_TARGET_FPS 追加 (0 = .bad ヘッダ値を自動使用)
 *   - CFG_FRAME_MS 廃止 → FPS 指定に統一
 *   - CFG_AUDIO_OUTPUT_MODE に 2 (SDM) を追加
 *   - CFG_AUDIO_SR を 0 (WAV ヘッダ自動) に変更
 *   - OSD 表示項目 (CPU/FPS/VU) の個別 ON/OFF 追加
 */

 #ifndef CONFIG_H
 #define CONFIG_H
 
 /* ---- I2C (OLED) ------------------------------------------ */
 #define CFG_I2C_PORT    I2C_NUM_0
 #define CFG_I2C_SDA     8
 #define CFG_I2C_SCL     9
 #define CFG_I2C_FREQ_HZ 1000000
 
 /* ---- SSD1306 128x64 -------------------------------------- */
 #define CFG_OLED_ADDR   0x3D
 #define CFG_PHYS_W      128
 #define CFG_PHYS_H      64
 #define CFG_PAGES       8
 #define CFG_DISP_W      128
 #define CFG_DISP_H      64
 #define CFG_X_OFFSET    0
 #define CFG_Y_OFFSET    0
 
 /* ---- BadCodec video -------------------------------------- */
 #define CFG_VIDEO_W     128
 #define CFG_VIDEO_H     64
 
 /* ---- フレームレート指定 -----------------------------------
  *
  * CFG_TARGET_FPS
  *   0  = .bad ファイルヘッダの fps 値を使用 (推奨・デフォルト)
  *   N  = 整数 fps を強制 (例: 30, 24, 15)
  *
  * フレーム間隔 ms は実行時に 1000/fps で自動計算される。
  * ---------------------------------------------------------- */
 /* fps × 100 で指定 (小数点第2位対応)
  * 2997=29.97fps 2500=25.00fps 3000=30.00fps 2400=24.00fps */
 #define CFG_TARGET_FPS_100  2398   /* 25.00fps */
 #define CFG_TARGET_FPS      0      /* 0=CFG_TARGET_FPS_100 を使用 */
 
 /* ---- Audio 出力モード ------------------------------------
  *
  * CFG_AUDIO_OUTPUT_MODE
  *   0 = SINGLE  シングルエンド PWM (GPIO_P のみ)
  *   1 = BTL     差動 PWM (GPIO_P 正相 + GPIO_N 逆相)
  *   2 = SDM     ハードウェア Sigma-Delta 変調 (GPIO_P のみ)
  *
  * --- SINGLE 接続 ---
  *   GPIO10 → R=10kΩ → C=100nF → GND  (LPF fc≈160Hz)
  *                    → C=10μF  → アンプ入力
  *
  * --- BTL 接続 ---
  *   GPIO10 (正相) → R=10kΩ → BTL アンプ IN+
  *   GPIO3  (逆相) → R=10kΩ → BTL アンプ IN−
  *
  *   BTL ホワイトノイズ対策:
  *     GPIO3 は UART0_RX 共用のため USB 接続時にノイズが増加する。
  *     sdkconfig.defaults に以下を追加してUARTログを抑制すること:
  *       CONFIG_ESP_CONSOLE_NONE=y
  *       CONFIG_LOG_DEFAULT_LEVEL_NONE=y
  *     また両ピンと GND 間に C=100nF を追加するとキャリア低減できる。
  *
  * --- SDM 接続 (最高 S/N 推奨) ---
  *   GPIO10 → R=1kΩ → C=47nF → GND  (LPF fc≈3.4kHz)
  *                  → C=10μF → アンプ入力
  *
  *   SDM の特長:
  *     - ESP32 ハードウェア SDM ペリフェラル使用
  *     - CPU 使用率ほぼ 0%、FPS への影響なし
  *     - キャリア ~312kHz → LPF で完全除去
  *     - BTL ホワイトノイズがない (シングルエンド)
  *     - 理論 SNR ≈ 84 dB (OSR=8x PCM 相当)
  * ---------------------------------------------------------- */
 #define CFG_AUDIO_OUTPUT_MODE   0  /* 0=SINGLE  1=BTL */
 #define CFG_AUDIO_PIN_P         7  /* 正相 / SDM 出力 GPIO */
 #define CFG_AUDIO_PIN_N         10   /* 逆相 GPIO (BTL 時のみ) */

 /* ---- 並列出力ピン (最大4本, 0=無効) --------------------
  * 各ピン独立 R=10kΩ → 合流 → LPF → アンプ
  * スピーカー直結: C=100μF カップリング
  * 例: GPIO10/4/5/6 の4本並列 → 電流4倍
  * -------------------------------------------------------- */
 #define CFG_AUDIO_PIN_P2  10   /* 並列ピン2 (0=無効, 例:4) */
 #define CFG_AUDIO_PIN_P3  0   /* 並列ピン3 (0=無効, 例:5) */
 #define CFG_AUDIO_PIN_P4  0
    /* 並列ピン4 (0=無効, 例:6) */
 
 /* サンプルレート: 0 = WAV ヘッダ値を自動使用 (推奨)
  * 非ゼロの場合はこの値で強制上書きする。           */
 #define CFG_AUDIO_SR    0U   /* 0 = auto from WAV header */
 
 /* Volume: 0=mute, 256=full scale
  * 歪みが出る場合は 200 程度に下げる。              */
 #define CFG_AUDIO_VOL   255U
 
 /* ---- Source ---------------------------------------------- */
 #define CFG_SOURCE_FLASH    1
 /* #define CFG_SOURCE_SD   1 */
 #ifdef CFG_SOURCE_SD
 #  define CFG_SD_FILE       "/sdcard/output.bad"
 #  define CFG_AUDIO_SD_FILE "/sdcard/adpcm4.wav"
 #endif
 
 /* ---- Video / Audio 有効化 -------------------------------- */
 #define CFG_VIDEO_ENABLE    1
 #define CFG_AUDIO_ENABLE    1
 
 /* ---- OSD 表示設定 ----------------------------------------
  *
  * 各項目を個別に ON/OFF できます。
  * 表示レイアウト (下段から):
  *   最下段: CPU xx.x%
  *   その上: FPS xx.x
  *   その上: VU [バーグラフ]
  *
  * CFG_OSD_CPU  1 = CPU 使用率表示
  * CFG_OSD_FPS  1 = FPS 表示
  * CFG_OSD_VU   1 = VU メーター表示 (15fps 更新)
  * ---------------------------------------------------------- */
 #define CFG_OSD_CPU     1   /* 1=表示  0=非表示 */
 #define CFG_OSD_FPS     1   /* 1=表示  0=非表示 */
 #define CFG_OSD_VU      1   /* 1=表示  0=非表示 */
 
 /* VU 更新間隔 (ms) — 1/15 秒 ≈ 66ms */
 #define CFG_OSD_VU_INTERVAL_MS  33U
 
  /* ---- OSD 表示位置 (Y座標, 8px単位) -----------------------
  * 各行は 8px フォントで表示。CFG_PHYS_H=64 のとき
  * 有効範囲: 0〜56 (56=最下段)
  * ---------------------------------------------------------- */
 #define OSD_LINE_1 40
 #define OSD_LINE_2 48
 #define OSD_LINE_3 56

 #define CFG_OSD_Y_CPU   OSD_LINE_1   /* CPU 使用率行 Y座標 */
 #define CFG_OSD_Y_FPS   OSD_LINE_2   /* FPS 行 Y座標       */
 #define CFG_OSD_Y_VU    OSD_LINE_3  /* VU バー行 Y座標    */

 #endif /* CONFIG_H */
 
