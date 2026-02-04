# BadCodec v0.02
Low-Spec Microcontroller Oriented Monochrome Video Codec

---

## 1. 概要

BadCodec は、**1bit（2値）モノクロ動画専用の可逆圧縮コーデック**です。  
低スペックマイコン環境での再生を目的として設計されています。

本コーデックは以下を重視しています。

- 低スペックマイコンでも再生可能  
- デコード処理が単純  
- デコード時のメモリ使用量が少ない  
- 動きの少ない映像で高い圧縮率

Python 実装は**リファレンス実装**であり、  
実際の組み込み用途では C 等への移植を想定しています。

---

## 2. 特徴（Features）

- 1bit（2値）モノクロ専用  
- 8×8 ブロック単位処理  
- ブロック差分・RLE・生データのハイブリッド圧縮  
- フレームごとに最小サイズ方式を自動選択  
- 低RAM・低ROM環境向け  
- SPI Flash 等からの逐次読み出し再生を想定  

---

## 3. 入力データ形式（Input Requirements）

### 3.1 ファイル形式

- BMP  
- 非圧縮  
- 1bit モノクロ（白黒2値）

内部で以下の変換を行う。

Image.open(file).convert("1")

### 3.2 画像サイズ

- 幅：8 の倍数  
- 高さ：8 の倍数  

BadCodec は画像を 8×8 ピクセルのブロックに分割して処理する。

### 3.3 ファイル並び

BMP の連番ファイルであること。

frame_0001.bmp
frame_0002.bmp
frame_0003.bmp

### 3.4 非対応入力

- PNG / JPEG  
- カラー画像  
- グレースケール 8bit  
- 縦横が 8 の倍数でない画像

---

## 4. 出力形式

- 拡張子：`.bad`  
- 独自バイナリ形式

---

## 5. ファイル構造

### 5.1 ヘッダ

| Offset | Size | 内容 |
|------|------|-----|
|0x00|3|"Bad"|
|0x03|2|Version|
|0x05|2|Width|
|0x07|2|Height|
|0x09|1|Block Size (8)|

---

## 6. 採用している圧縮方式

BadCodec は **ハイブリッド圧縮方式**を採用する。

### 6.1 Block Stream（ブロック差分圧縮）

フレームを 8×8 ブロックに分割し、前フレーム同位置ブロックと比較。

使用されるブロック命令：

- SKIP_BLOCK  
- INVERT_BLOCK  
- FILL_BLOCK  
- SHIFT_BLOCK  
- RLE_BLOCK  
- MASTER_BLOCK  

### 6.2 RLE Frame

フレーム全体を 1bit RLE 圧縮。

[1bit 色][7bit 長さ]
0x00 = 終端

### 6.3 Master Frame

フレーム全体を bitpack した生データ。

### 6.4 方式選択

Block Stream と RLE Frame を比較 → 小さい方
↓
その結果と Master Frame を比較 → 小さい方

---

## 7. 使い方（Usage）

### 7.1 エンコード

python badcodec.py -t e -p frames -s 1 -e 6572 -o movie.bad

### 7.2 デコード

python badcodec.py -t d -i movie.bad -p out_frames

---

## 8. 計算コスト

### 8.1 エンコード

- ブロック単位で探索処理を行うため**時間がかかる**
- リアルタイム用途には向かない

### 8.2 デコード

- 分岐とメモリコピー主体  
- 低負荷

---

## 9. 圧縮実測結果

条件：

- 2値（1bit）  
- 解像度：128×64  
- フレーム数：6572 frames  

RAW サイズ：

128 × 64 = 8192 pixel
8192 / 8 = 1024 bytes / frame
1024 × 6572 = 6,729,728 bytes

エンコード後：

997182 bytes
(test260203_02.bad, Feb 3 17:29)

圧縮率：

997182 / 6729728 ≒ 0.148
約 14.8 %

---

## 10. 想定用途

- Bad Apple 等の白黒動画  
- マイコン + OLED / LCD  
- SPI Flash 動画再生

---
