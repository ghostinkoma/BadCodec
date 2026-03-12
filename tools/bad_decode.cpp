/**

- @file    bad_decode.cpp
- @brief   BadCodec v0.5.1 - Decoder implementation
- @version 0.5.1  (Protocol: 510)
- @date    2026-03-10
- @license Non-Commercial Use Only  ghostinkoma@gmail.com
- 
- Design constraints (全プラットフォーム共通):
- - NO malloc / free
- - NO floating point
- - NO division  (>>3 / &7 で /8 と %8 を代替)
- - Stack local variables: 最大 ~32 bytes (AVR safe)
- - <stdint.h> のみ依存
    */

#include “bad_decode.h”

/* ============================================================

- Internal: memset / memcpy
- <string.h> 非依存
- ============================================================ */

static void bad_memset(uint8_t *dst, uint8_t val, uint16_t len)
{
while (len–) { *dst++ = val; }
}

static void bad_memcpy(uint8_t *dst, const uint8_t *src, uint16_t len)
{
while (len–) { *dst++ = *src++; }
}

/* ============================================================

- Internal: Fletcher-16  (SPEC.md 3-3節)
- ============================================================ */

static uint16_t fletcher16(const uint8_t *data, uint8_t len)
{
uint8_t s1 = 0U;
uint8_t s2 = 0U;
while (len–) {
s1 = (uint8_t)(s1 + *data++);
s2 = (uint8_t)(s2 + s1);
}
return (uint16_t)(((uint16_t)s2 << 8U) | s1);
}

/* ============================================================

- Internal: read helpers
- ctx->read() を通じてのみデータにアクセスする
- ============================================================ */

static uint8_t bad_read1(bad_ctx_t *ctx)
{
uint8_t b = 0xFFU;
ctx->read(ctx->stream_offset, &b, 1U);
ctx->stream_offset++;
return b;
}

static uint16_t bad_readn(bad_ctx_t *ctx, uint8_t *buf, uint16_t len)
{
uint16_t n = ctx->read(ctx->stream_offset, buf, len);
ctx->stream_offset += n;
return n;
}

/* ============================================================

- Internal: gram ビット操作ヘルパー
- 
- gram は1次元ビット配列。
- bit_idx = y * width + x  （行方向・リトルエンディアン）
- byte    = bit_idx >> 3
- bit pos = bit_idx &  7
- 
- NOTE: BAD_STATIC_INLINE は既に “static inline” を含むため
- ```
    ここでは static を付けない
  ```
- ============================================================ */

BAD_STATIC_INLINE uint8_t gp_get(const uint8_t *buf, uint16_t bit_idx)
{
return (buf[bit_idx >> 3U] >> (bit_idx & 7U)) & 1U;
}

BAD_STATIC_INLINE void gp_set(uint8_t *buf, uint16_t bit_idx, uint8_t val)
{
if (val)
buf[bit_idx >> 3U] |=  (uint8_t)(1U << (bit_idx & 7U));
else
buf[bit_idx >> 3U] &= ~(uint8_t)(1U << (bit_idx & 7U));
}

/* ============================================================

- Internal: ブロック座標ヘルパー
- ============================================================ */

BAD_STATIC_INLINE uint8_t blocks_x(const bad_ctx_t *ctx)
{
return (uint8_t)(ctx->width  >> 3U);
}

BAD_STATIC_INLINE uint8_t blocks_y(const bad_ctx_t *ctx)
{
return (uint8_t)(ctx->height >> 3U);
}

/**

- ブロック (bx, by) 内ピクセル (px, py) の global bit_idx を返す
  */
  BAD_STATIC_INLINE uint16_t blk_bit(const bad_ctx_t *ctx,
  uint8_t bx, uint8_t by,
  uint8_t px, uint8_t py)
  {
  return (uint16_t)(((uint16_t)(by << 3U) + py) * ctx->width
  + ((uint16_t)(bx << 3U) + px));
  }

/* ============================================================

- Internal: フレームバッファ操作
- ============================================================ */

static void fill_frame(bad_ctx_t *ctx, uint8_t color)
{
uint16_t sz = BAD_GRAM_SIZE(ctx->width, ctx->height);
bad_memset(ctx->gram, color ? 0xFFU : 0x00U, sz);
}

static void swap_to_prev(bad_ctx_t *ctx)
{
uint16_t sz = BAD_GRAM_SIZE(ctx->width, ctx->height);
bad_memcpy(ctx->prev, ctx->gram, sz);
}

/* ============================================================

- Internal: ブロック単位操作
- ============================================================ */

static void fill_block(bad_ctx_t *ctx,
uint8_t bx, uint8_t by, uint8_t color)
{
uint8_t px, py;
for (py = 0U; py < BAD_BLOCK_SIZE; py++)
for (px = 0U; px < BAD_BLOCK_SIZE; px++)
gp_set(ctx->gram, blk_bit(ctx, bx, by, px, py), color);
}

static void copy_block(bad_ctx_t *ctx, uint8_t bx, uint8_t by)
{
uint8_t px, py;
for (py = 0U; py < BAD_BLOCK_SIZE; py++) {
for (px = 0U; px < BAD_BLOCK_SIZE; px++) {
uint16_t idx = blk_bit(ctx, bx, by, px, py);
gp_set(ctx->gram, idx, gp_get(ctx->prev, idx));
}
}
}

static void invert_block(bad_ctx_t *ctx, uint8_t bx, uint8_t by)
{
uint8_t px, py;
for (py = 0U; py < BAD_BLOCK_SIZE; py++) {
for (px = 0U; px < BAD_BLOCK_SIZE; px++) {
uint16_t idx = blk_bit(ctx, bx, by, px, py);
gp_set(ctx->gram, idx, gp_get(ctx->prev, idx) ^ 1U);
}
}
}

/* ============================================================

- Internal: MASTER_BLOCK decode  (SPEC.md 6-7節)
- 
- 後続8バイト = ブロック1行8ピクセル × 8行
- リトルエンディアン・行方向  bit0 = 左端ピクセル
- ============================================================ */

static bad_result_t decode_master_block(bad_ctx_t *ctx,
uint8_t bx, uint8_t by)
{
uint8_t row, col, byte_buf;
for (row = 0U; row < BAD_BLOCK_SIZE; row++) {
uint8_t n = (uint8_t)ctx->read(ctx->stream_offset, &byte_buf, 1U);
if (n != 1U) return BAD_ERR_DATA;
ctx->stream_offset++;
for (col = 0U; col < BAD_BLOCK_SIZE; col++)
gp_set(ctx->gram,
blk_bit(ctx, bx, by, col, row),
(byte_buf >> col) & 1U);
}
return BAD_OK;
}

/* ============================================================

- Internal: SHIFT_BIT decode  (SPEC.md 6-3節)
- 
- local stack: rows[8] = 8 bytes ← AVR safe
- 
- パディング: 移動前ブロックの端ビット値で埋める
- +X (右移動) : 左端を右端ビット(bit7)で埋める
- -X (左移動) : 右端を左端ビット(bit0)で埋める
- +Y (下移動) : 上端を下端行で埋める
- -Y (上移動) : 下端を上端行で埋める
- ============================================================ */

static void decode_shift_bit(bad_ctx_t *ctx,
uint8_t bx, uint8_t by,
uint8_t op)
{
uint8_t rows[BAD_BLOCK_SIZE]; /* 8 bytes stack */
uint8_t r, c;

```
/* 1. prev ブロックを rows[] に展開（1行1バイト、bit0=左端） */
for (r = 0U; r < BAD_BLOCK_SIZE; r++) {
    rows[r] = 0U;
    for (c = 0U; c < BAD_BLOCK_SIZE; c++) {
        if (gp_get(ctx->prev, blk_bit(ctx, bx, by, c, r)))
            rows[r] |= (uint8_t)(1U << c);
    }
}

/* 2. X 方向シフト */
uint8_t mag_x = BAD_SHIFT_MAG_X(op);
if (mag_x > 0U) {
    uint8_t sh = mag_x;
    if (BAD_SHIFT_SIGN_X(op) == 0U) {
        /* +X 右移動: 左端を右端ビット(bit7)で埋める */
        for (r = 0U; r < BAD_BLOCK_SIZE; r++) {
            uint8_t pad = (rows[r] & 0x80U) ? (uint8_t)(0xFFU << (8U - sh)) : 0x00U;
            rows[r] = (uint8_t)((rows[r] << sh) | (pad >> (8U - sh)));
        }
    } else {
        /* -X 左移動: 右端を左端ビット(bit0)で埋める */
        for (r = 0U; r < BAD_BLOCK_SIZE; r++) {
            uint8_t pad = (rows[r] & 0x01U) ? (uint8_t)(0xFFU >> (8U - sh)) : 0x00U;
            rows[r] = (uint8_t)((rows[r] >> sh) | (pad << (8U - sh)));
        }
    }
}

/* 3. Y 方向シフト */
uint8_t mag_y = BAD_SHIFT_MAG_Y(op);
if (mag_y > 0U) {
    uint8_t i;
    if (BAD_SHIFT_SIGN_Y(op) == 0U) {
        /* +Y 下移動: 上端を下端行で埋める */
        uint8_t edge_row = rows[BAD_BLOCK_SIZE - 1U];
        for (i = BAD_BLOCK_SIZE - 1U; i >= mag_y; i--)
            rows[i] = rows[i - mag_y];
        for (i = 0U; i < mag_y; i++)
            rows[i] = edge_row;
    } else {
        /* -Y 上移動: 下端を上端行で埋める */
        uint8_t edge_row = rows[0U];
        uint8_t limit    = (uint8_t)(BAD_BLOCK_SIZE - mag_y);
        for (i = 0U; i < limit; i++)
            rows[i] = rows[i + mag_y];
        for (i = limit; i < BAD_BLOCK_SIZE; i++)
            rows[i] = edge_row;
    }
}

/* 4. gram に書き戻す */
for (r = 0U; r < BAD_BLOCK_SIZE; r++)
    for (c = 0U; c < BAD_BLOCK_SIZE; c++)
        gp_set(ctx->gram,
               blk_bit(ctx, bx, by, c, r),
               (rows[r] >> c) & 1U);
```

}

/* ============================================================

- Internal: RLE_BLOCK decode  (SPEC.md 6-5節)
- 
- 8走査パターン × 4ラン → 64 ピクセル描画
- local stack: rle[3] + run[4] ← AVR safe
- ============================================================ */

static bad_result_t decode_rle_block(bad_ctx_t *ctx,
uint8_t bx, uint8_t by,
uint8_t op)
{
/* 後続3バイト */
uint8_t rle[3];
if (bad_readn(ctx, rle, 3U) != 3U) return BAD_ERR_DATA;

```
/* 4ラン長を復元（各6bit）*/
uint8_t run[4];
run[0] = (uint8_t)BAD_RLE_UNPACK_R0(rle[0]);
run[1] = (uint8_t)BAD_RLE_UNPACK_R1(rle[0], rle[1]);
run[2] = (uint8_t)BAD_RLE_UNPACK_R2(rle[1], rle[2]);
run[3] = (uint8_t)BAD_RLE_UNPACK_R3(rle[2]);

/* 走査パターン解析 (SPEC.md 6-5節)
 * pattern = [scan_dir(1)][start_y(1)][start_x(1)]
 *   scan_dir : 0=水平優先  1=垂直優先
 *   start_y  : 0=上(y0)    1=下(y7)
 *   start_x  : 0=左(x0)    1=右(x7)
 */
uint8_t pat      = BAD_RLE_PATTERN(op);
uint8_t color    = BAD_RLE_START_COLOR(op);
uint8_t scan_dir = (pat >> 2U) & 1U;
int8_t  sx       = (pat & 1U) ? 7 : 0;
int8_t  sy       = (pat & 2U) ? 7 : 0;
int8_t  dx       = (pat & 1U) ? -1 : 1;
int8_t  dy       = (pat & 2U) ? -1 : 1;

/* ピクセル描画 */
uint8_t ri     = 0U;
uint8_t remain = run[0];
uint8_t drawn  = 0U;

uint8_t outer;
for (outer = 0U; outer < BAD_BLOCK_SIZE && drawn < BAD_BLOCK_PIXELS; outer++) {
    uint8_t inner;
    for (inner = 0U; inner < BAD_BLOCK_SIZE && drawn < BAD_BLOCK_PIXELS; inner++) {
        uint8_t px, py;
        if (scan_dir == 0U) {
            /* 水平優先: outer=行, inner=列 */
            py = (uint8_t)(sy + (int8_t)outer * dy);
            px = (uint8_t)(sx + (int8_t)inner * dx);
        } else {
            /* 垂直優先: outer=列, inner=行 */
            px = (uint8_t)(sx + (int8_t)outer * dx);
            py = (uint8_t)(sy + (int8_t)inner * dy);
        }

        gp_set(ctx->gram, blk_bit(ctx, bx, by, px, py), color);
        drawn++;

        if (remain > 0U) remain--;
        if (remain == 0U) {
            color ^= 1U;
            ri++;
            if (ri < 4U) {
                remain = run[ri];
            }
            /* remain==0 → 残りを現在色で埋め続ける */
        }
    }
}
return BAD_OK;
```

}

/* ============================================================

- Internal: BLOCK_STREAM decode  (SPEC.md 7-7節)
- 
- FOR ネスト禁止。FOR 直後の1命令を N 回繰り返す。
- ============================================================ */

static bad_result_t decode_block_stream(bad_ctx_t *ctx)
{
uint8_t  nbx   = blocks_x(ctx);
uint16_t total = (uint16_t)nbx * (uint16_t)blocks_y(ctx);
uint16_t ptr   = 0U;
bad_result_t ret;

```
while (ptr < total) {
    uint8_t op = bad_read1(ctx);

    /* ---- FOR ------------------------------------------- */
    if (BAD_IS_FOR(op)) {
        uint8_t repeat   = (uint8_t)BAD_FOR_COUNT(op);
        uint8_t inner_op = bad_read1(ctx);
        uint8_t fi;

        for (fi = 0U; fi < repeat && ptr < total; fi++) {
            uint8_t bx = (uint8_t)(ptr % nbx);
            uint8_t by = (uint8_t)(ptr / nbx);

            if (BAD_IS_SKIP(inner_op)) {
                uint8_t cnt = (uint8_t)BAD_SKIP_COUNT(inner_op);
                while (cnt-- && ptr < total) {
                    copy_block(ctx,
                               (uint8_t)(ptr % nbx),
                               (uint8_t)(ptr / nbx));
                    ptr++;
                }
            } else if (BAD_IS_INVERT(inner_op)) {
                uint8_t cnt = (uint8_t)BAD_INVERT_COUNT(inner_op);
                while (cnt-- && ptr < total) {
                    invert_block(ctx,
                                 (uint8_t)(ptr % nbx),
                                 (uint8_t)(ptr / nbx));
                    ptr++;
                }
            } else if (BAD_IS_FILL(inner_op)) {
                uint8_t color = BAD_FILL_COLOR(inner_op);
                uint8_t cnt   = (uint8_t)BAD_FILL_COUNT(inner_op);
                while (cnt-- && ptr < total) {
                    fill_block(ctx,
                               (uint8_t)(ptr % nbx),
                               (uint8_t)(ptr / nbx),
                               color);
                    ptr++;
                }
            } else if (BAD_IS_SHIFT(inner_op)) {
                decode_shift_bit(ctx, bx, by, inner_op);
                ptr++;
            } else {
                return BAD_ERR_DATA;
            }
        }
        continue;
    }

    /* ---- 通常ブロック命令 ------------------------------ */
    uint8_t bx = (uint8_t)(ptr % nbx);
    uint8_t by = (uint8_t)(ptr / nbx);

    if (BAD_IS_SKIP(op)) {
        uint8_t cnt = (uint8_t)BAD_SKIP_COUNT(op);
        while (cnt-- && ptr < total) {
            copy_block(ctx,
                       (uint8_t)(ptr % nbx),
                       (uint8_t)(ptr / nbx));
            ptr++;
        }

    } else if (BAD_IS_INVERT(op)) {
        uint8_t cnt = (uint8_t)BAD_INVERT_COUNT(op);
        while (cnt-- && ptr < total) {
            invert_block(ctx,
                         (uint8_t)(ptr % nbx),
                         (uint8_t)(ptr / nbx));
            ptr++;
        }

    } else if (BAD_IS_FILL(op)) {
        uint8_t color = BAD_FILL_COLOR(op);
        uint8_t cnt   = (uint8_t)BAD_FILL_COUNT(op);
        while (cnt-- && ptr < total) {
            fill_block(ctx,
                       (uint8_t)(ptr % nbx),
                       (uint8_t)(ptr / nbx),
                       color);
            ptr++;
        }

    } else if (BAD_IS_SHIFT(op)) {
        decode_shift_bit(ctx, bx, by, op);
        ptr++;

    } else if (BAD_IS_RLE(op)) {
        ret = decode_rle_block(ctx, bx, by, op);
        if (ret != BAD_OK) return ret;
        ptr++;

    } else if (BAD_IS_MASTER_BLOCK(op)) {
        ret = decode_master_block(ctx, bx, by);
        if (ret != BAD_OK) return ret;
        ptr++;

    } else {
        return BAD_ERR_DATA;
    }
}
return BAD_OK;
```

}

/* ============================================================

- Internal: RLE_FRAME decode  (SPEC.md 7-6節)
- 
- bit7=色, bit6-0=ラン長(1-127)  0x00=終端
- ============================================================ */

static bad_result_t decode_rle_frame(bad_ctx_t *ctx)
{
uint16_t total_px = (uint16_t)(ctx->width) * (uint16_t)(ctx->height);
uint16_t drawn    = 0U;
uint16_t bit_idx  = 0U;

```
while (drawn < total_px) {
    uint8_t b = bad_read1(ctx);
    if (b == 0x00U) return BAD_ERR_DATA;

    uint8_t  color = BAD_RLEFRAME_COLOR(b);
    uint8_t  len   = (uint8_t)BAD_RLEFRAME_LEN(b);
    if (len == 0U)  return BAD_ERR_DATA;

    uint8_t i;
    for (i = 0U; i < len && drawn < total_px; i++, drawn++, bit_idx++)
        gp_set(ctx->gram, bit_idx, color);
}
return BAD_OK;
```

}

/* ============================================================

- Internal: MASTER_FRAME decode  (SPEC.md 7-5節)
- ============================================================ */

static bad_result_t decode_master_frame(bad_ctx_t *ctx)
{
uint16_t sz = BAD_GRAM_SIZE(ctx->width, ctx->height);
if (bad_readn(ctx, ctx->gram, sz) != sz) return BAD_ERR_DATA;
return BAD_OK;
}

/* ============================================================

- Internal: FRAME_DELIMITER をスキップしてオペコードを返す
- ============================================================ */

static uint8_t read_frame_op(bad_ctx_t *ctx)
{
uint8_t op;
do { op = bad_read1(ctx); } while (op == BAD_OP_FRAME_DELIMITER);
return op;
}

/* ============================================================

- Public API: bad_init
- ============================================================ */

bad_result_t bad_init(bad_ctx_t *ctx)
{
if (ctx == NULL)         return BAD_ERR_ARG;
if (ctx->read == NULL)   return BAD_ERR_ARG;
if (ctx->gram == NULL)   return BAD_ERR_ARG;
if (ctx->prev == NULL)   return BAD_ERR_ARG;
if (ctx->buf_size == 0U) return BAD_ERR_ARG;

```
/* ヘッダー読み出し (stack 19 bytes) */
uint8_t  hdr[BAD_HEADER_SIZE];
uint16_t n = ctx->read(0U, hdr, BAD_HEADER_SIZE);
if (n < BAD_HEADER_SIZE) return BAD_ERR_HDR;

/* hdr_size (LE uint16, offset 0-1) */
uint16_t hdr_size = (uint16_t)(((uint16_t)hdr[1] << 8U) | hdr[0]);
if (hdr_size != BAD_HEADER_SIZE) return BAD_ERR_HDR;

/* Fletcher-16 (LE uint16, offset 2-3) */
uint16_t stored_chk = (uint16_t)(((uint16_t)hdr[3] << 8U) | hdr[2]);
uint16_t calc_chk   = fletcher16(&hdr[4], BAD_HEADER_BODY_SIZE);
if (stored_chk != calc_chk) return BAD_ERR_HDR;

/* Magic "Bad" (offset 4-6) */
if (hdr[4] != (uint8_t)'B' ||
    hdr[5] != (uint8_t)'a' ||
    hdr[6] != (uint8_t)'d') return BAD_ERR_MAGIC;

/* ヘッダー本体フィールド (全 LE uint16)
 * hdr[4..6]  = "Bad"          (3 bytes)
 * hdr[7..8]  = version        (2 bytes)
 * hdr[9..10] = colors         (2 bytes)
 * hdr[11..12]= width          (2 bytes)  ← index 11,12 ... wait
 *
 * ヘッダー本体 offset 0 = hdr[4]
 * body[0..2]  = "Bad"
 * body[3..4]  = version  → hdr[7..8]
 * body[5..6]  = colors   → hdr[9..10]
 * body[7..8]  = width    → hdr[11..12] ... hdr[4+7]=hdr[11], hdr[4+8]=hdr[12]
 * body[9..10] = height   → hdr[13..14]
 * body[11..12]= blksize  → hdr[15..16] ... wait let me recount
 *
 * hdr[4]  = body[0] = 'B'
 * hdr[5]  = body[1] = 'a'
 * hdr[6]  = body[2] = 'd'
 * hdr[7]  = body[3] = ver_lo
 * hdr[8]  = body[4] = ver_hi
 * hdr[9]  = body[5] = colors_lo
 * hdr[10] = body[6] = colors_hi
 * hdr[11] = body[7] = width_lo
 * hdr[12] = body[8] = width_hi
 * hdr[13] = body[9] = height_lo
 * hdr[14] = body[10]= height_hi
 * hdr[15] = body[11]= blksize_lo
 * hdr[16] = body[12]= blksize_hi
 * hdr[17] = body[13]= total_frames_lo
 * hdr[18] = body[14]= total_frames_hi
 */
ctx->width        = (uint16_t)(((uint16_t)hdr[12] << 8U) | hdr[11]);
ctx->height       = (uint16_t)(((uint16_t)hdr[14] << 8U) | hdr[13]);
ctx->total_frames = (uint16_t)(((uint16_t)hdr[18] << 8U) | hdr[17]);

/* バッファサイズ確認 */
uint16_t need = BAD_GRAM_SIZE(ctx->width, ctx->height);
if (ctx->buf_size < need) return BAD_ERR_MEM;

/* 内部状態初期化 */
ctx->stream_offset = BAD_HEADER_SIZE;
ctx->current_frame = 0U;
ctx->initialized   = 1U;

/* prev をゼロクリア（初回フレームの前フレームは全黒）*/
bad_memset(ctx->prev, 0x00U, need);

return BAD_OK;
```

}

/* ============================================================

- Public API: bad_next_frame
- ============================================================ */

bad_result_t bad_next_frame(bad_ctx_t *ctx)
{
if (ctx == NULL || ctx->initialized == 0U) return BAD_ERR_ARG;
if (ctx->current_frame >= ctx->total_frames) return BAD_EOF;

```
swap_to_prev(ctx);

uint8_t      op  = read_frame_op(ctx);
bad_result_t ret = BAD_OK;

switch (op) {

case BAD_OP_SKIP_FRAME:
    bad_memcpy(ctx->gram, ctx->prev,
               BAD_GRAM_SIZE(ctx->width, ctx->height));
    break;

case BAD_OP_FILL_BLACK:
    fill_frame(ctx, 0U);
    break;

case BAD_OP_FILL_WHITE:
    fill_frame(ctx, 1U);
    break;

case BAD_OP_INVERT_PREV: {
    uint16_t sz = BAD_GRAM_SIZE(ctx->width, ctx->height);
    uint16_t i;
    for (i = 0U; i < sz; i++)
        ctx->gram[i] = (uint8_t)~ctx->prev[i];
    break;
}

case BAD_OP_MASTER_FRAME:
    ret = decode_master_frame(ctx);
    break;

case BAD_OP_RLE_FRAME:
    ret = decode_rle_frame(ctx);
    break;

case BAD_OP_BLOCK_STREAM:
    ret = decode_block_stream(ctx);
    break;

case BAD_OP_MASTER_BLOCK:
    /* フレーム先頭 MASTER_BLOCK (SPEC.md 第8章) */
    ret = decode_master_block(ctx, 0U, 0U);
    break;

case BAD_OP_EXT_PREFIX:
    /* 拡張命令: 現バージョン未定義 → サブコマンド読み捨て */
    (void)bad_read1(ctx);
    bad_memcpy(ctx->gram, ctx->prev,
               BAD_GRAM_SIZE(ctx->width, ctx->height));
    break;

default:
    ret = BAD_ERR_DATA;
    break;
}

if (ret == BAD_OK) {
    ctx->current_frame++;
    if (ctx->current_frame >= ctx->total_frames)
        ret = BAD_EOF;
}

return ret;
```

}

/* ============================================================

- Public API: bad_rewind
- ============================================================ */

bad_result_t bad_rewind(bad_ctx_t *ctx)
{
if (ctx == NULL || ctx->initialized == 0U) return BAD_ERR_ARG;
ctx->stream_offset = BAD_HEADER_SIZE;
ctx->current_frame = 0U;
bad_memset(ctx->prev, 0x00U,
BAD_GRAM_SIZE(ctx->width, ctx->height));
return BAD_OK;
}

/* ============================================================

- Public API: bad_seek
- ============================================================ */

bad_result_t bad_seek(bad_ctx_t *ctx, uint16_t frame_no)
{
if (ctx == NULL || ctx->initialized == 0U) return BAD_ERR_ARG;
if (frame_no >= ctx->total_frames)          return BAD_EOF;

```
bad_result_t ret = bad_rewind(ctx);
if (ret != BAD_OK) return ret;

while (ctx->current_frame < frame_no) {
    ret = bad_next_frame(ctx);
    if (ret != BAD_OK && ret != BAD_EOF) return ret;
}
return BAD_OK;
```

}

/* ============================================================

- Public API: bad_result_str
- Flash 節約したい場合はリンクしないこと
- ============================================================ */

const char *bad_result_str(bad_result_t result)
{
switch (result) {
case BAD_OK:        return “BAD_OK”;
case BAD_BUSY:      return “BAD_BUSY”;
case BAD_EOF:       return “BAD_EOF”;
case BAD_ERR_HDR:   return “BAD_ERR_HDR”;
case BAD_ERR_MAGIC: return “BAD_ERR_MAGIC”;
case BAD_ERR_DATA:  return “BAD_ERR_DATA”;
case BAD_ERR_MEM:   return “BAD_ERR_MEM”;
case BAD_ERR_ARG:   return “BAD_ERR_ARG”;
default:            return “BAD_ERR_UNKNOWN”;
}
}