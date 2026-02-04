BadCodec Ver0.002
Low-Spec Microcontroller Oriented Monochrome Video Codec
1. 概要
BadCodec は、1bit モノクロ動画専用の可逆圧縮コーデックです。
低スペックマイコン環境での再生を目的として設計されており、
RAM 使用量を抑える
デコード処理を単純化する
動きの少ない映像で高圧縮率を得る
ことを目標としています。
本リポジトリに含まれる Python 実装は リファレンス実装であり、
実運用では C などへの移植を想定しています。
2. 入力データ形式（Input Requirements）
エンコード時に使用する画像は、以下の条件をすべて満たす必要があります。
2.1 ファイル形式
BMP 形式
非圧縮
1bit モノクロ（白黒2値）
内部では以下の処理が行われます。
Image.open(file).convert('1')
2.2 画像サイズ
幅：8 の倍数
高さ：8 の倍数
BadCodec は画像を 8×8 ピクセル単位のブロックに分割して処理します。
block_x = width / 8
block_y = height / 8
2.3 ファイル並び
BMP の 連番ファイルである必要があります。
例：
frames/
 ├ frame_0001.bmp
 ├ frame_0002.bmp
 ├ frame_0003.bmp
2.4 非対応入力
PNG / JPEG
カラー画像
グレースケール 8bit
縦横が 8 の倍数でない画像
3. 出力データ形式
拡張子：.bad
独自バイナリ形式
4. ヘッダ構造
Offset	Size	内容
0x00	3	"Bad"
0x03	2	Version
0x05	2	Width
0x07	2	Height
0x09	1	Block Size (8)
5. 採用している圧縮方式
BadCodec は ハイブリッド圧縮方式です。
各フレームについて以下の 3 方式を生成し、
最もサイズの小さい方式を採用します。
Block Stream
RLE Frame
Master Frame
5.1 Block Stream（ブロック差分圧縮）
フレームを 8×8 ブロックに分割し、
各ブロックを前フレームの同位置ブロックと比較します。
使用される方式：
SKIP_BLOCK
前フレームと完全一致。
INVERT_BLOCK
前フレームを 0↔1 反転したものと一致。
FILL_BLOCK
ブロック全体が白または黒。
SHIFT_BLOCK
前フレームブロックを
X または Y 方向に ±1～7 ピクセル移動した結果と一致。
RLE_BLOCK
ブロック内部を 1bit RLE 圧縮。
MASTER_BLOCK
64bit 生データ。
5.2 RLE Frame
フレーム全体を 1bit RLE 圧縮。
[色1bit][長さ7bit]
0xFF = 同色 +127
0x00 = 終端
5.3 Master Frame
フレーム全体をそのまま bitpack。
5.4 方式選択
Block Stream vs RLE Frame → 小さい方
↓
その結果 vs Master Frame → 小さい方
6. 使い方（Usage）
6.1 エンコード
python badcodec.py -t e -p frames -s 1 -e 300 -o movie.bad
主なオプション
オプション	内容
-t e	エンコード
-p PATH	BMP フォルダ
-s N	開始番号
-e N	終了番号
-o FILE	出力 bad
-n STR	接頭辞（省略時 frame_）
6.2 デコード
python badcodec.py -t d -i movie.bad -p out_frames
7. 計算コストと注意
7.1 エンコードは時間がかかる
各ブロックで
シフト探索
RLE パターン探索
を行うため、エンコードは非常に重い処理です。
リアルタイム用途には向きません。
7.2 デコードは軽量
分岐とメモリコピー主体で、
マイコン実装を想定しています。
8. 想定用途
Bad Apple 等の白黒動画
マイコン + OLED/LCD 再生
SPI Flash から逐次再生