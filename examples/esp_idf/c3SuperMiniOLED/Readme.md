# BadCodecPlayer

**ESP32-C3 Super Mini (OLED 72x40 ビルトイン) で BadCodec 動画を再生する**

BadCodec v0.6.0 / Protocol 514 / SPEC rev.19  
PlatformIO + ESP-IDF フレームワーク

---

## ハードウェア

```
ESP32-C3 Super Mini OLED ビルトインモジュール
  SSD1306 OLED  物理 128x64 / 有効表示域 72x40
  I2C SDA : GPIO5
  I2C SCL : GPIO6
  ADDR    : 0x3C
  速度    : 1 MHz
```

**追加配線は不要です。** OLED はモジュールに内蔵されています。

---

## ディレクトリ構成

```
BadCodecPlayer/
├── platformio.ini          PlatformIO / ESP-IDF 設定
├── sdkconfig.defaults      ESP-IDF カーネル設定
├── README.md               このファイル
├── include/
│   ├── config.h            ★ GPIO / FPS / 解像度の一元管理
│   ├── bad_decode.h        BadCodec v0.6.0 デコーダ ヘッダ
│   ├── ssd1306_drv.h       SSD1306 ドライバ ヘッダ
│   └── bad_data.h          ★ 動画データ (→ 手順3 で差し替える)
└── src/
    ├── main.c              再生ロジック・app_main
    ├── ssd1306_drv.c       I2C + framebuffer + gram 変換
    └── bad_decode.cpp      BadCodec デコーダ実装
```

---

## 事前準備

### 必要なツール

```bash
# Python パッケージ
pip install Pillow numpy

# PlatformIO (VS Code 拡張または CLI)
pip install platformio
```

---

## 手順

### 1. 動画フレームを BMP に変換する

72x40 白黒 BMP の連番ファイルを用意する。

```bash
# ffmpeg の例: mp4 → 72x40 白黒 BMP 連番
ffmpeg -i input.mp4 \
    -vf "scale=72:40:force_original_aspect_ratio=decrease,\
         pad=72:40:(ow-iw)/2:(oh-ih)/2,\
         format=gray" \
    -pix_fmt monob \
    frames/frame_%04d.bmp

# フレーム数を確認
ls frames/ | wc -l
```

> **解像度は必ず 72x40 にすること。**  
> 8 の倍数でなければならない (72 = 9×8 / 40 = 5×8 ✓)

---

### 2. BadCodec でエンコードする

```bash
# Codec.py のある場所に移動
cd /path/to/BadCodec/tools

# エンコード
python3 Codec.py -t e \
    -p ./frames \
    -n frame_ \
    -s 0001 \
    -e 6572 \
    -o output.bad

# 実行例の出力:
# BadCodec v0.6.0 Encoder  [8 cores]  Phase 1/2: Encoding
# ...
# Done. Saved to output.bad (972,249 bytes, 14.45% of raw  6.922x)
#   Frame FOR merge: 972,398B → 972,230B  (saved 168B)
```

**オプション説明:**

| オプション | 説明 |
|-----------|------|
| `-p` | BMP フレームのディレクトリ |
| `-n` | ファイル名プレフィックス (例: `frame_` → `frame_0001.bmp`) |
| `-s` | 開始フレーム番号 |
| `-e` | 終了フレーム番号 |
| `-o` | 出力 .bad ファイル名 |

---

### 3. C ヘッダファイルを生成する

```bash
python3 Codec.py -t c \
    -i output.bad \
    -H bad_data.h
```

**生成されるファイルの内容:**

```c
#define BAD_DATA_WIDTH    72U
#define BAD_DATA_HEIGHT   40U
#define BAD_DATA_FRAMES   6572U
#define BAD_DATA_SIZE     972249UL

#ifdef __AVR__
const uint8_t bad_data[] PROGMEM = {
#else
const uint8_t bad_data[] = {
#endif
    0x13, 0x00, 0xXX, ...
};
```

---

### 4. ヘッダをプロジェクトに配置する

```bash
cp bad_data.h /path/to/BadCodecPlayer/include/bad_data.h
```

---

### 5. ビルドして書き込む

```bash
cd BadCodecPlayer

# ビルド + 書き込み
pio run --target upload

# シリアルモニタ (動作確認)
pio device monitor
```

**正常時の出力:**

```
I (xxx) BadCodec: BadCodecPlayer v0.6.0  Protocol 514
I (xxx) BadCodec: Board: ESP32-C3 Super Mini  OLED: 72x40 builtin
I (xxx) BadCodec: Source: Flash
I (xxx) BadCodec: OK  72x40  6572 frames  ~34fps
```

---

## Flash 容量

```
ESP32-C3 Mini Flash: 4MB
  ファームウェア本体: ~200KB
  bad_data (972KB 動画): ~972KB
  合計: ~1.2MB → 4MB に余裕で収まる ✓

動画が 3MB を超える場合は SD カードを使用すること。
```

---

## SD カードから再生する場合

### 1. config.h を編集する

```c
/* include/config.h */
/* #define CFG_SOURCE_FLASH  1 */   /* ← コメントアウト */
#define CFG_SOURCE_SD     1          /* ← 有効化 */
```

### 2. SD カードに動画を置く

```
SD カード (FAT32)
└── output.bad
```

### 3. SD 接続ピン (デフォルト)

SD モジュールを別途接続する必要があります。  
`config.h` の `CFG_SD_*` を使用する SD モジュールのピンに合わせて変更してください。

---

## フレームレート調整

`include/config.h` を編集する:

```c
#define CFG_FRAME_MS  29   /* 約 34fps */
/* #define CFG_FRAME_MS  33 */  /* 約 30fps */
/* #define CFG_FRAME_MS  41 */  /* 約 24fps */
```

元動画の FPS に合わせて調整すること。

---

## 画面上のピクセル配置

```
SSD1306 物理パネル 128x64:

┌────────────────────────────────────────┐  y=0
│                                        │
│                                        │  y=11 (CFG_Y_OFFSET=12 の直前)
│   ┌──────────────────────────────┐    │  y=12  ← 動画 y=0
│   │                              │    │
│   │    Bad Apple!! 72x40 動画    │    │
│   │                              │    │
│   └──────────────────────────────┘    │  y=51  ← 動画 y=39
│                                        │
└────────────────────────────────────────┘  y=63
x=0                              x=27+72=99    x=127

CFG_X_OFFSET = 27  (左右余白)
CFG_Y_OFFSET = 12  (上下余白)
```

---

## トラブルシューティング

| 症状 | 確認事項 |
|------|---------|
| 画面が映らない | GPIO5/6 の I2C 接続。`pio device monitor` でログ確認 |
| 全点灯点滅 | `bad_init` 失敗。`bad_data.h` のヘッダを確認 |
| 映像が画面端にずれる | `CFG_X_OFFSET` / `CFG_Y_OFFSET` を調整 |
| コマ落ち | `CFG_FRAME_MS` を大きくする |
| Flash 容量不足 | `CFG_SOURCE_SD` に切り替える |

---

## ライセンス

Non-Commercial Use Only  
Contact: ghostinkoma@gmail.com
