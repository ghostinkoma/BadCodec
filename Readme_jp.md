# BadCodec

**8ビットマイコン向け2値モノクロ動画コーデック**

[![Version](https://img.shields.io/badge/version-0.5.5-blue)](CHANGELOG.md)
[![Protocol](https://img.shields.io/badge/protocol-055-lightgrey)](SPEC.md)
[![License](https://img.shields.io/badge/license-Non--Commercial-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-pre--release-orange)]()

> **Pre-release**: エンコード/デコード整合性テストが完了するまで 0.x.x を維持します。

---

## 概要

BadCodec は LGT8F328（RAM 2KB）等の極低スペックマイコン、
ESP32などのマイコン上で動作モノクロ2値動画をリアルタイム再生するためのコーデックです。

**設計思想：**

```
- 1バイト命令体系   オペコード1バイトで即命令確定
- 前フレームバッファ1枚のみ   動的メモリ確保なし
- ビット演算のみ   乗除算・浮動小数点なし
- Fletcher-16 チェックサム   加算のみ・RAM 2B
- read コールバック   Flash/SD/SPIFFS/LittleFS を抽象化
```

---

## 実績圧縮率

```
例:
Bad Apple!! 128x64 / 6572フレーム / 2値モノクロ / ロスレス

  非圧縮     : 6.42 MB  (100%)
  BadCodec   : 0.93 MB  (14.4%)  6.92x

制約条件:
  符号化なし (Huffman / 算術符号 等 不使用)
  辞書圧縮なし (LZ77 / LZW 等 不使用)
  LGT8F328 (2KB RAM) でリアルタイムデコード可能

```

---

## ターゲットプラットフォーム

| ターゲット | CPU | RAM | 推奨解像度 |
|-----------|-----|-----|----------|
| LGT8F328（ミニマム） | AVR互換 32MHz | 2KB | 128×64 |
| ESP32-Cx（標準） | RISC-V 160MHz | 400KB | 320×240以下 |
| RP2350（マキシマム） | Cortex-M33 150MHz | 520KB | 640×480以下 |

---

## 命令体系（Protocol 514）

### フレームレベル命令

| 命令 | オペコード | 説明 |
|------|-----------|------|
| RLE_FRAME | 0x30-0x37 | フレーム全体RLE・8走査パターン |
| SKIP_FRAME | 0x39 | 前フレームをそのまま維持 |
| FRAME_FILL_BLACK | 0x3A | 全画素を黒で塗りつぶす |
| FRAME_FILL_WHITE | 0x3B | 全画素を白で塗りつぶす |
| BLOCK_STREAM | 0x3C | ブロック命令列 |
| DELTA_FRAME | 0x3D | XOR差分フレームRLE・8走査パターン |
| MASTER_FRAME | 0x3E | rawビットデータ |
| INVERT_PREV_FRAME | 0x3F | 前フレーム全ビット反転 |

### ブロックレベル命令

| 命令 | オペコード | サイズ | 説明 |
|------|-----------|--------|------|
| SKIP_BLOCK | 0x80-0xBF | 1B | 前フレームコピー (1-64ブロック) |
| FILL_BLOCK | 0x30-0x37 | 1B | 単色塗りつぶし (1-4ブロック) |
| BLOCK_INVERT | 0x00-0x1F | 1B | 前フレーム反転 (1-32ブロック) |
| SHIFT_BIT | 0x40-0x7F | 1B | ±3ドット微小移動差分 |
| RLE_BLOCK_4 | 0x20-0x2F | 4B | 8方向RLE・4ラン |
| RLE_BLOCK_8 | 0x38-0x3B,0x3D-0x3E | 7B | 拡張RLE・8ラン |
| XOR_BLOCK | 0x3F | 2+NB | prev XOR差分RLE |
| MASTER_BLOCK | 0x3C | 9B | rawビットデータ |
| FOR | 0xC0-0xFF | 2B | 次命令を4-65回繰り返す (最小repeat=4) |

---

## インストール

```bash
git clone https://github.com/ghostinkoma/BadCodec.git
cd BadCodec
pip install Pillow numpy
```

---

## 使い方

### エンコード

```bash
python3 tools/Codec.py -t e \
  -p ./frames \
  -n frame_ \
  -s 0001 \
  -e 6572 \
  -o output.bad
```

### デコード

```bash
python3 tools/Codec.py -t d \
  -i output.bad \
  -p ./out \
  -n frame_ \
  -s 0001
```

### オプション

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `-t` | 必須 | `e`=エンコード / `d`=デコード |
| `-p` | 必須 | BMPディレクトリ |
| `-n` | `frame_` | ファイル名接頭辞 |
| `-s` | `0001` | 開始フレーム番号 |
| `-e` | 必須(encode) | 終了フレーム番号 |
| `-o` | `output.bad` | 出力ファイル名 |
| `-i` | 必須(decode) | 入力.badファイル |

---

## マイコン向けデコーダ

`tools/bad_decode.h` / `tools/bad_decode.cpp` を使用します。

```c
#include "bad_decode.h"

static uint8_t   gram[BAD_GRAM_SIZE(128, 64)];
static uint8_t   prev[BAD_GRAM_SIZE(128, 64)];
static bad_ctx_t ctx;

uint16_t my_read(bad_addr_t off, uint8_t *buf, uint16_t len) {
    for (uint16_t i = 0; i < len; i++)
        buf[i] = pgm_read_byte(VIDEO_DATA + off + i);
    return len;
}

void setup() {
    ctx.read     = my_read;
    ctx.gram     = gram;
    ctx.prev     = prev;
    ctx.buf_size = sizeof(gram);
    bad_init(&ctx);
}

void loop() {
    bad_result_t r = bad_next_frame(&ctx);
    if (r == BAD_OK)  display_write(ctx.gram);
    if (r == BAD_EOF) bad_rewind(&ctx);
}
```

---

## ファイルフォーマット

```
[19バイト ヘッダー]
  2B: ヘッダーサイズ (固定 19)
  2B: Fletcher-16 チェックサム
  3B: マジックナンバー "Bad"
  2B: プロトコルバージョン (514)
  2B: カラー数 (2)
  2B: 画像幅
  2B: 画像高さ
  2B: ブロックサイズ (8)
  2B: 総フレーム数

[フレームデータ]
  FRAME_DELIMITER(0x38) + フレーム命令 の繰り返し
```

詳細は [SPEC.md](SPEC.md) を参照。

---

## 開発状況

```
[済] コーデック仕様確定 (SPEC.md rev.18)
[済] Python エンコーダ/デコーダ (Protocol 055)
[済] マルチCPU並列エンコード
[済] Self-Verify 機構
[済] Cデコーダ (bad_decode.h / bad_decode.cpp)
[済] FOR最適化 再帰的最適計算 (rev.18)
[済] DELTA_FRAME (rev.16)
[  ] Cデコーダ実機検証
[  ] エンコード/デコード整合性テストスイート
```

---

## ライセンス

非商用利用は自由。商用利用はお問い合わせください。

**Contact:** ghostinkoma@gmail.com  
詳細は [LICENSE](LICENSE) を参照。
