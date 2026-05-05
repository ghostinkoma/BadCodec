# BadCodec

**8ビットマイコン向け2値モノクロ動画コーデック**

[![Version](https://img.shields.io/badge/version-0.5.1-blue)](CHANGELOG.md)
[![Protocol](https://img.shields.io/badge/protocol-514-lightgrey)](SPEC.md)
[![License](https://img.shields.io/badge/license-Non--Commercial-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-pre--release-orange)]()

# BadCodecPlayer for ESP32-C3 Super Mini

**BadCodec v0.6.0 / Protocol 514 / ESP-IDF v5.0.2**

[English](#english) | [日本語](#japanese)

---

<a name="japanese"></a>
## 日本語

### 概要

ESP32-C3 Super Mini + SSD1306 128×64 OLED で、IMA-ADPCM 音声付き BadCodec 動画をリアルタイム再生するプレーヤーです。外部 DAC・WiFi・BT を一切使わず、抵抗とコンデンサのみで音声出力します。

**対応コンテンツ例:**
- Bad Apple!! (30fps)
- Take On Me MV (25fps / 29.97fps)

---

### ハードウェア構成

| 部品 | 仕様 |
|---|---|
| MCU | ESP32-C3 Super Mini (160MHz, RISC-V, 320KB RAM, 4MB Flash) |
| ディスプレイ | SSD1306 128×64 OLED (I2C) |
| 音声出力 | LEDC PWM → RC LPF → アンプ (最大4ピン並列) |

#### GPIO 割り当て

| GPIO | 機能 |
|---|---|
| 8 | I2C SDA (OLED) |
| 9 | I2C SCL (OLED) |
| 10 | 音声出力 P (正相・必須) |
| 3 | 音声出力 N (BTL 逆相時のみ) |
| 4 | ボタン PLAY/PAUSE |
| 5 | ボタン OSD ON/OFF |
| 6 | ボタン VOL+ |
| 7 | ボタン VOL− |

#### 音声出力回路

**SINGLE モード (mode=0, 推奨):**
```
GPIO10 → R=10kΩ → C=100nF→GND
                → C=10μF → アンプ入力
```

**並列出力 (音圧増強):**
```
GPIO10 → R=10kΩ ┐
GPIO4  → R=10kΩ ├→ C=100nF→GND / C=10μF→アンプ
GPIO5  → R=10kΩ ┘
```
各ピンに必ず個別の R=10kΩ を入れること。並列直結は禁止。

**BTL モード (mode=1):**
```
GPIO10 (正相) → R=10kΩ → BTL アンプ IN+
GPIO3  (逆相) → R=10kΩ → BTL アンプ IN−
```

---

### ソフトウェア構成

```
src/
├── main.c          プレーヤー本体 (v3.1.0)
├── adpcm_drv.c/h   IMA-ADPCM デコーダ + LEDC PWM 音声ドライバ (v4.0.0)
├── bad_decode.c/h  BadCodec デコーダ (v0.5.1 / Protocol 514)
├── draw.c/h        描画モジュール + OSD レイヤー
├── button.c/h      ボタン入力 (GPIO 内部プルアップ、チャタリング除去)
├── ssd1306_drv.c/h SSD1306 I2C ドライバ
├── font.h          8×8 等幅フォント
└── config.h        全設定値
tools/
└── Codec.py        BadCodec エンコーダ / デコーダ / C ヘッダ出力
```

---

### config.h 主要設定

```c
/* I2C */
#define CFG_I2C_FREQ_HZ   1000000      /* 1MHz (動作実績あり) */

/* フレームレート (fps × 100 で指定、小数点第2位まで対応) */
#define CFG_TARGET_FPS_100  2500       /* 25.00fps (PAL) */
// 2997 = 29.97fps (NTSC)
// 3000 = 30.00fps
// 2400 = 24.00fps (映画)

/* 音声出力モード */
#define CFG_AUDIO_OUTPUT_MODE  0       /* 0=SINGLE 1=BTL */
#define CFG_AUDIO_PIN_P        10      /* 正相 GPIO (必須) */
#define CFG_AUDIO_PIN_N        3       /* 逆相 GPIO (BTL 時) */

/* 並列出力ピン (0=無効) */
#define CFG_AUDIO_PIN_P2  0            /* 例: 4 */
#define CFG_AUDIO_PIN_P3  0            /* 例: 5 */
#define CFG_AUDIO_PIN_P4  0            /* 例: 6 */

/* 音量 0=無音 256=フルスケール */
#define CFG_AUDIO_VOL  220U

/* OSD 表示 */
#define CFG_OSD_CPU  1
#define CFG_OSD_FPS  1
#define CFG_OSD_VU   1
```

---

### アーキテクチャ

#### 音声同期方式
```
gptimer ISR (16kHz)
  → LEDC duty 更新 (IRAM 配置、ISR から安全)
  → s_isr_samples カウント
  → samples / spf == 新フレーム番号 → xSemaphoreGiveFromISR()
  → player_task が xSemaphoreTake() でフレームタイミングを受け取る
```

フレーム間隔は `adpcm_set_frame_us()` で μs 単位設定。  
29.97fps = 33366μs → 誤差 0.003%。

#### 音声バッファ設計 (v4.0.0)
```
adpcm_decode_task:
  フェーズ1 (prefill): リングが 75% 以上になるまで連続デコード
                        → s_prefill_done=1
  adpcm_init():         s_prefill_done を待機してから gptimer 起動
  フェーズ2 (通常):     ring_full なら vTaskDelay(1) で待機しながら継続
```
- ライターは常に `adpcm_decode_task` のみ（二重ライター競合なし）
- ブロック境界を必ず守る → テンポずれなし

#### OSD レイヤー設計
```
映像バッファ g_fb[1024]      ← ssd1306_blit_gram() が書く
OSD レイヤー s_osd_layer[1024] ← OSD 専用

描画順:
  g_fb ← clear + blit_gram
  s_osd_layer ← clear + OSD 描画
  g_fb ^= s_osd_layer  (INVERT 重畳)
  ssd1306_flush()
```
ボタン押下タイミングで画面が乱れない構造。

---

### ボタン操作

| ボタン | GPIO | 機能 | チャタリング |
|---|---|---|---|
| PLAY/PAUSE | 4 | 映像+音声同時停止/再開、中央 PAUSE 点滅 | 150ms |
| OSD | 5 | CPU/FPS/VU 表示を 3bit 順次トグル | 150ms |
| VOL+ | 6 | 音量+ (16段階)、1.5秒バー表示 | 80ms |
| VOL− | 7 | 音量− (16段階)、1.5秒バー表示 | 80ms |

ボタン未接続でも動作（常時 HIGH = 非押下）。内部プルアップ使用。

---

### エンコード

```bash
# BMP → .bad エンコード
python3 tools/Codec.py -t e -p ./frames -n frame_ -s 0001 -e 6572 -o output.bad

# .bad → C ヘッダ
python3 tools/Codec.py -t c -i output.bad -H bad_data.h

# 音声変換 (IMA-ADPCM WAV → C ヘッダ)
python3 tools/Wave2adpcmH.py -i audio.wav -o adpcm4.h
```

---

### ビルド

```bash
# PlatformIO (推奨)
pio run --target upload --environment esp32c3-supermini

# ESP-IDF
idf.py build flash monitor
```

**メモリ使用量 (実測):**
```
RAM:   43.3% (142KB / 320KB)
Flash: 87.6% (3.6MB / 4MB)
```

---

### 既知のバグ / 制限事項

- SDM モード (mode=2) は廃止。ISR から Flash 関数呼び出し不可の根本問題により削除。
- FPS 自動検出は未実装。`config.h` の `CFG_TARGET_FPS_100` を手動設定すること。
- `bad_ctx_t` に fps フィールドがないため自動取得不可（将来版で対応予定）。

---


### ライセンス

BadCodec: Non-Commercial Use Only (ghostinkoma@gmail.com)

---

<a name="english"></a>
## English

### Overview

A real-time BadCodec video player with IMA-ADPCM audio for ESP32-C3 Super Mini + SSD1306 128×64 OLED. No external DAC, WiFi, or Bluetooth required — audio output via resistor + capacitor only.

**Tested content:**
- Bad Apple!! (30fps)
- Take On Me MV (25fps / 29.97fps)

---

### Hardware

| Component | Spec |
|---|---|
| MCU | ESP32-C3 Super Mini (160MHz, RISC-V, 320KB RAM, 4MB Flash) |
| Display | SSD1306 128×64 OLED (I2C @ 1MHz) |
| Audio | LEDC PWM → RC LPF → amplifier (up to 4 parallel pins) |

#### GPIO Assignment

| GPIO | Function |
|---|---|
| 8 | I2C SDA (OLED) |
| 9 | I2C SCL (OLED) |
| 10 | Audio P (positive, required) |
| 3 | Audio N (BTL negative only) |
| 4 | Button PLAY/PAUSE |
| 5 | Button OSD toggle |
| 6 | Button VOL+ |
| 7 | Button VOL− |

#### Audio Circuit

**SINGLE mode (mode=0, recommended):**
```
GPIO10 → R=10kΩ → C=100nF→GND
                → C=10μF → amp input
```

**Parallel output (louder):**
```
GPIO10 → R=10kΩ ┐
GPIO4  → R=10kΩ ├→ C=100nF→GND / C=10μF→amp
GPIO5  → R=10kΩ ┘
```
Each pin **must** have its own series resistor. Never connect in parallel directly.

**BTL mode (mode=1):**
```
GPIO10 (+) → R=10kΩ → BTL amp IN+
GPIO3  (−) → R=10kΩ → BTL amp IN−
```

---

### config.h Key Settings

```c
/* Frame rate: fps × 100 (supports 2 decimal places) */
#define CFG_TARGET_FPS_100  2997   // 29.97fps (NTSC)
// 2500 = 25.00fps (PAL)
// 3000 = 30.00fps
// 2400 = 24.00fps (film)

/* Audio output mode */
#define CFG_AUDIO_OUTPUT_MODE  0   // 0=SINGLE 1=BTL

/* Parallel output pins (0=disabled) */
#define CFG_AUDIO_PIN_P2  0        // e.g. 4
#define CFG_AUDIO_PIN_P3  0        // e.g. 5
#define CFG_AUDIO_PIN_P4  0        // e.g. 6

/* Volume: 0=mute, 256=full scale */
#define CFG_AUDIO_VOL  220U
```

---

### Architecture

#### Audio Sync
```
gptimer ISR @ sample_rate (e.g. 16kHz)
  → ledc_set_duty() [IRAM, safe from ISR]
  → count samples
  → samples / spf == new frame → xSemaphoreGiveFromISR()
  → player_task receives timing via xSemaphoreTake()
```

Frame interval set via `adpcm_set_frame_us()` for μs precision.  
29.97fps = 33366μs → error 0.003%.

#### Double-writer bug fix (v4.0.0)
- `prefill_ring()` abolished — caused IMA-ADPCM block boundary violation
- `adpcm_decode_task` is the sole writer to the ring buffer
- Phase 1 (prefill) runs inside the task before gptimer starts
- Phase 2 (normal) continues indefinitely

#### OSD Layer
```
g_fb[1024]          ← video frame
s_osd_layer[1024]   ← OSD only

Draw order:
  g_fb ← clear + blit_gram
  s_osd_layer ← clear + draw OSD
  g_fb ^= s_osd_layer   (INVERT blend)
  ssd1306_flush()
```
No screen glitch on button press.

---

### Button Controls

| Button | GPIO | Function | Debounce |
|---|---|---|---|
| PLAY/PAUSE | 4 | Toggle video+audio, blink PAUSE | 150ms |
| OSD | 5 | Toggle CPU/FPS/VU display (3-bit) | 150ms |
| VOL+ | 6 | Volume up (16 steps), show bar 1.5s | 80ms |
| VOL− | 7 | Volume down (16 steps), show bar 1.5s | 80ms |

Works without buttons connected (internal pull-up, open = not pressed).

---

### Encoding

```bash
# BMP frames → .bad
python3 tools/Codec.py -t e -p ./frames -n frame_ -s 0001 -e 6572 -o output.bad

# .bad → C header
python3 tools/Codec.py -t c -i output.bad -H bad_data.h

# Audio: WAV → IMA-ADPCM C header
python3 tools/Wave2adpcmH.py -i audio.wav -o adpcm4.h
```

---

### Build

```bash
# PlatformIO (recommended)
pio run --target upload --environment esp32c3-supermini

# ESP-IDF
idf.py build flash monitor
```

**Memory usage:**
```
RAM:   43.3% (142KB / 320KB)
Flash: 87.6% (3.6MB / 4MB)
```

---

### Known Issues / Limitations

- SDM mode (mode=2) removed — ISR cannot call Flash functions.
- FPS auto-detection not implemented. Set `CFG_TARGET_FPS_100` manually.
- `bad_ctx_t` has no fps field (planned for future version).


---

### License

BadCodec: Non-Commercial Use Only (ghostinkoma@gmail.com)
