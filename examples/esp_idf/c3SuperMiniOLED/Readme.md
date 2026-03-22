# BadCodecPlayer

**ESP32-C3 Suoer Mini + SSD1306 OLED で BadCodec 動画を再生する PlatformIO プロジェクト**

BadCodec v0.6.0 / Protocol 514 / SPEC rev.19

---

## ハードウェア構成

```
ESP32-C3 Mini
  GPIO5 → SDA
  GPIO6 → SCL
  3.3V  → VCC
  GND   → GND

SSD1306 OLED (128x64 I2C)
  物理解像度: 128x64
  動画表示域: 72x40 (画面中央に配置)
  I2C アドレス: 0x3C
  I2C 速度: 1 MHz
```

---

## ディレクトリ構成

```
BadCodecPlayer/
├── platformio.ini          PlatformIO 設定
├── sdkconfig.defaults      ESP-IDF 設定
├── README.md               このファイル
├── include/
│   ├── bad_player_config.h ★ ハードウェア設定 (GPIO等はここで変更)
│   ├── bad_decode.h        BadCodec デコーダ ヘッダ
│   ├── ssd1306.h           SSD1306 ドライバ ヘッダ
│   └── bad_data.h          ★ 動画データ (要生成・差し替え)
└── src/
    ├── main.c              メインアプリ
    ├── ssd1306.c           SSD1306 ドライバ実装
    └── bad_decode.cpp      BadCodec デコーダ実装
```

---

## クイックスタート

### 1. 動画フレームを用意する

72x40 px の白黒 BMP 連番ファイルを用意する。

```bash
# ffmpeg で mp4 → 72x40 白黒 BMP 連番に変換する例
ffmpeg -i input.mp4 \
       -vf "scale=72:40,format=gray,binarize=thresh=128" \
       -pix_fmt monob \
       frames/frame_%04d.bmp
```

### 2. BadCodec でエンコードする

```bash
# エンコード (72x40 で必ずエンコードすること)
python3 tools/Codec.py -t e \
    -p ./frames \
    -n frame_ \
    -s 0001 \
    -e 6572 \
    -o output.bad

# 結果例:
# Done. Saved to output.bad (972,249 bytes, 14.45% of raw  6.922x)
```

### 3. C ヘッダファイルを生成する

```bash
# C ヘッダ生成 (-H オプションでファイル名を指定)
python3 tools/Codec.py -t c \
    -i output.bad \
    -H bad_data.h

# 生成されるファイル内容:
#   #define BAD_DATA_WIDTH   72U
#   #define BAD_DATA_HEIGHT  40U
#   #define BAD_DATA_FRAMES  6572U
#   const uint8_t bad_data[] = { 0x13, 0x00, ... };
```

### 4. ヘッダファイルをプロジェクトに配置する

```bash
cp bad_data.h BadCodecPlayer/include/bad_data.h
```

### 5. ビルドして書き込む

```bash
cd BadCodecPlayer
pio run --target upload

# シリアルモニタで確認
pio device monitor
```

---

## 設定変更

`include/bad_player_config.h` を編集する。

```c
/* GPIO */
#define BAD_I2C_SDA    5    /* 変更可 */
#define BAD_I2C_SCL    6    /* 変更可 */

/* 動画解像度 (エンコード時と一致させること) */
#define BAD_VIDEO_WIDTH   72
#define BAD_VIDEO_HEIGHT  40

/* フレームレート */
#define BAD_FRAME_MS   29   /* 約34fps。元動画のFPSに合わせて調整 */

/* 読み込み元 */
#define BAD_SOURCE_FLASH 1   /* Flash から読む (デフォルト) */
/* #define BAD_SOURCE_SD 1 */ /* SD カードから読む場合はこちらを有効化 */
```

---

## SD カードから再生する場合

`bad_player_config.h` で `BAD_SOURCE_SD` を有効化する。

```c
/* #define BAD_SOURCE_FLASH 1 */  /* ← コメントアウト */
#define BAD_SOURCE_SD    1         /* ← 有効化 */
```

SD カードのルートに `output.bad` を置く。

```
SD カード
└── output.bad
```

---

## Flash 容量の目安

```
ESP32-C3 Mini フラッシュ: 4MB
アプリ本体 (ファームウェア): 約 200KB
bad_data.h (972KB 動画):    約 972KB
合計:                       約 1.2MB → 4MB に収まる ✓

より大きな動画の場合は SD カード再生を使用すること。
```

---

## 動画サイズと OLED の関係

```
SSD1306 物理解像度: 128 x 64 px
BadCodec 動画域:     72 x 40 px

配置:
  ┌────────────────────────────────┐ 128px
  │           (余白 12px)          │
  │   ┌──────────────────────┐    │
  │   │                      │    │ ← 72x40 動画
  │   │     動画表示域        │    │
  │   └──────────────────────┘    │
  │           (余白 12px)          │
  └────────────────────────────────┘

xOffset = 27  (画面左右余白)
yOffset = 12  (画面上下余白)
```

---

## トラブルシューティング

| 症状 | 原因 | 対処 |
|------|------|------|
| 画面が全点灯して点滅 | bad_init 失敗 | bad_data.h の内容を確認 |
| 画像がずれる | オフセット設定ミス | BAD_X_OFFSET / BAD_Y_OFFSET を調整 |
| コマ落ちする | フレームレートが速い | BAD_FRAME_MS を大きくする |
| フラッシュ容量不足 | 動画が大きすぎる | SD カードを使用する |

---

## ライセンス

Non-Commercial Use Only  
Contact: ghostinkoma@gmail.com
