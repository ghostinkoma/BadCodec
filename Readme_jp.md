# BadCodec

**8ビットマイコン向け2値モノクロ動画コーデック**

[![Version](https://img.shields.io/badge/version-0.5.1-blue)](CHANGELOG.md)
[![License](https://img.shields.io/badge/license-Non--Commercial-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-pre--release-orange)]()

> ⚠️ **Pre-release**: エンコード/デコードの完全な整合性テストが完了するまで 0.x.x として管理します。

-----

## 概要

BadCodecは、極低スペックマイコン上でモノクロ2値動画をリアルタイム再生することを目的として設計されたコーデックです。

デコーダの設計思想：

- **1バイト命令体系** ── オペコード1バイトで即座に命令確定
- **前フレームバッファ1枚のみ** ── 動的メモリ確保なし
- **ビット演算のみ** ── 乗除算・浮動小数点なし
- **ルックアップテーブル不要** ── 最小Flashフットプリント
- **Fletcher-16チェックサム** ── 2バイト・加算のみ・マイコン最適解

-----

## ターゲットプラットフォーム

|ターゲット         |CPU              |RAM  |推奨解像度    |後段圧縮     |
|--------------|-----------------|-----|---------|---------|
|LGT8F328（ミニマム）|AVR互換 32MHz      |2KB  |128×64   |不要       |
|ESP32-Cx（標準）  |RISC-V 160MHz    |400KB|320×240以下|静的Huffman|
|RP2350（マキシマム） |Cortex-M33 150MHz|520KB|640×480以下|DEFLATE相当|

-----

## 特徴

### 命令体系

```
フレーム命令（1バイト）:
  SKIP_FRAME       前フレームをそのまま維持
  FRAME_FILL_BLACK フレーム全体を黒で塗りつぶす
  FRAME_FILL_WHITE フレーム全体を白で塗りつぶす
  INVERT_PREV      前フレームを全ビット反転
  RLE_FRAME        フレーム全体RLE（可変長）
  MASTER_FRAME     フレーム全体rawデータ（固定長）
  BLOCK_STREAM     ブロック命令列

ブロック命令（1バイト）:
  SKIP_BLOCK    前フレームブロックをコピー（1〜64）
  FILL_BLOCK    単色塗りつぶし（黒/白、1〜4ブロック）
  BLOCK_INVERT  前フレームブロックを反転（1〜32）
  SHIFT_BIT     微小移動差分（±3ドット XY独立）
  RLE_BLOCK     8方向RLE（4バイト固定）
  MASTER_BLOCK  rawブロックデータ（9バイト固定）
  FOR           命令繰り返し（2〜65回）
```

### 圧縮効率（推定）

|コンテンツ             |圧縮率   |
|------------------|------|
|静止・メニュー           |1〜2%  |
|UIアニメーション         |5〜15% |
|スクロール             |10〜20%|
|ドット絵アニメーション       |15〜35%|
|Bad Apple!! 型シルエット|25〜50%|

-----

## インストール

```bash
git clone https://github.com/yourname/badcodec.git
cd badcodec
pip install Pillow numpy
```

-----

## 使い方

### エンコード

```bash
python badcodec.py -t e \
  -p ./frames \
  -n frame_ \
  -s 0001 \
  -e 6572 \
  -o output.bad
```

### デコード

```bash
python badcodec.py -t d \
  -i output.bad \
  -p ./out \
  -n frame_ \
  -s 0001
```

### オプション一覧

|オプション|デフォルト       |説明                  |
|-----|------------|--------------------|
|`-t` |必須          |`e`=エンコード / `d`=デコード|
|`-p` |必須          |BMPディレクトリ（入力/出力）    |
|`-n` |`frame_`    |ファイル名接頭辞            |
|`-s` |`0001`      |開始フレーム番号            |
|`-e` |必須（encode）  |終了フレーム番号            |
|`-o` |`output.bad`|出力ファイル名（encode）     |
|`-i` |必須（decode）  |入力.badファイル（decode）  |

### 入力フォーマット

```
<path>/<suffix><frame_number:04d>.bmp

例: ./frames/frame_0001.bmp 〜 ./frames/frame_6572.bmp
```

- モノクロ2値BMPのみ対応
- 解像度は幅・高さともに8の倍数であること

-----

## ファイルフォーマット

```
[19バイト ヘッダー]
  2バイト: ヘッダーサイズ
  2バイト: Fletcher-16チェックサム
  3バイト: マジックナンバー "Bad"
  2バイト: バージョン (510)
  2バイト: カラー数 (2)
  2バイト: 画像幅
  2バイト: 画像高さ
  2バイト: ブロックサイズ (8)
  2バイト: 総フレーム数

[フレームデータ]
  フレーム数分のフレーム命令列
```

詳細は <SPEC.md> を参照。

-----

## 開発状況

```
✅ コーデック仕様確定（SPEC.md rev.11）
✅ Pythonエンコーダ実装
✅ Pythonデコーダ実装
✅ マルチCPU並列エンコード
✅ フレームFOR最適化
✅ Self-Verify機構
⬜ エンコード/デコード整合性テスト
⬜ マイコン向けCデコーダ
⬜ 各ターゲット実機検証
```

-----

## ライセンス

非商用利用は自由です。商用利用については下記にお問い合わせください。

**Contact:** ghostinkoma@gmail.com

詳細は <LICENSE> を参照。