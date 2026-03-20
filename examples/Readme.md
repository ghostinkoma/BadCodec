# BadCodec v0.5.5 サンプル集

## ディレクトリ構成

```
examples/
├── arduino/
│   ├── BadCodecPlayer/          Arduino ライブラリ
│   │   ├── BadCodecPlayer.h
│   │   └── BadCodecPlayer.cpp
│   ├── from_flash/
│   │   └── from_flash.ino       ヘッダファイル (bad_data.h) から再生
│   └── from_sd/
│       └── from_sd.ino          SD カードから再生
├── esp_idf/
│   ├── from_flash/
│   │   └── main.c               bad_data.h から再生
│   └── from_sd/
│       └── main.c               SD カードから再生
├── rp2040/
│   ├── from_flash/
│   │   └── main.c               bad_data.h から再生
│   └── from_sd/
│       └── main.c               SD カード (FatFs) から再生
└── generic_c/
    └── main.c                   汎用C / Linux / macOS / Windows
```

---

## 共通準備

### 1. .bad ファイルの生成

```bash
python3 tools/Codec.py -t e \
  -p ./frames -n frame_ -s 0001 -e 6572 \
  -o output.bad
```

### 2. C ヘッダの生成 (Flash 再生用)

```bash
python3 tools/Codec.py -t c \
  -i output.bad -H bad_data.h
```

ヘッダ内の配列名は `-H` で指定したファイル名から自動生成されます。
`bad_data.h` → 配列名 `bad_data[]`

### 3. デコーダファイルの配置

以下の2ファイルを各プロジェクトにコピーしてください。

```
tools/bad_decode.h
tools/bad_decode.cpp
```

---

## Arduino

### ライブラリのインストール

`examples/arduino/BadCodecPlayer/` フォルダを
`~/Documents/Arduino/libraries/BadCodecPlayer/` にコピー。

### Flash 再生 (from_flash.ino)

```
from_flash/
├── from_flash.ino
├── bad_data.h        ← Codec.py -t c で生成
├── bad_decode.h      ← tools/ からコピー
└── bad_decode.cpp    ← tools/ からコピー
```

### SD カード再生 (from_sd.ino)

SD カードのルートに `output.bad` を置く。

```
from_sd/
├── from_sd.ino
├── bad_decode.h
└── bad_decode.cpp
```

### ディスプレイ接続 (SSD1306 128x64)

| 信号 | Arduino Uno |
|------|-------------|
| SDA  | A4          |
| SCL  | A5          |
| VCC  | 3.3V / 5V   |
| GND  | GND         |

---

## ESP-IDF

### Flash 再生 (esp_idf/from_flash)

```
from_flash/
├── CMakeLists.txt
├── main/
│   ├── CMakeLists.txt
│   ├── main.c
│   └── bad_data.h
└── components/
    └── bad_codec/
        ├── CMakeLists.txt
        ├── bad_decode.h
        └── bad_decode.cpp
```

**CMakeLists.txt (コンポーネント):**

```cmake
idf_component_register(
    SRCS "bad_decode.cpp"
    INCLUDE_DIRS "."
)
```

### SD カード再生 (esp_idf/from_sd)

`sdconfig.defaults` で SDMMC を有効化し、
SD カードに `output.bad` を配置する。

---

## RP2040 / RP2350

### Flash 再生 (rp2040/from_flash)

**CMakeLists.txt:**

```cmake
cmake_minimum_required(VERSION 3.13)
include(pico_sdk_import.cmake)
project(bad_player C CXX)
pico_sdk_init()
add_executable(bad_player main.c bad_decode.cpp)
target_link_libraries(bad_player pico_stdlib hardware_i2c)
pico_add_extra_outputs(bad_player)
```

### SD カード再生 (rp2040/from_sd)

FatFs ライブラリ:
https://github.com/carlk3/no-OS-FatFS-SD-SDIO-SPI-RPi-Pico

---

## 汎用C (Linux / macOS / Windows)

```bash
# Linux / macOS
gcc -std=c99 -O2 main.c ../bad_decode.cpp -lstdc++ -o badplay

# ヘッダファイルから再生
./badplay

# ファイルから再生
./badplay --file output.bad
```

ターミナルに ASCII アートで表示されます (30fps)。

---

## gram バッファのビット構造

```
1bit/pixel  LSB first  row-major

pixel(x,y) = (gram[y*width/8 + x/8] >> (x%8)) & 1

0=黒  1=白

BAD_GET_PIXEL(gram, x, y, width) マクロで取得可能
```

---

## フレームレート調整

```c
#define TARGET_FPS   30
#define FRAME_MS    (1000 / TARGET_FPS)   /* 33ms */
```

各プラットフォームの sleep 関数でフレーム間隔を調整してください。
エンコード時の FPS に合わせて変更できます。
