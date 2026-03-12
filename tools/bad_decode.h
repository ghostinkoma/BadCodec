/**

- @file    bad_decode.h
- @brief   BadCodec v0.5.1 - Decoder for 8/32-bit MCU
- @version 0.5.1  (Protocol: 510)
- @date    2026-03-10
- @license Non-Commercial Use Only  ghostinkoma@gmail.com
- 
- Supported platforms:
- ATmega328 / LGT8F328  Arduino (avr-gcc)
- ESP32-Cx              ESP-IDF (xtensa/riscv gcc)
- RP2040 / RP2350       PlatformIO (arm-none-eabi-gcc)
- 
- Design policy:
- - NO malloc / free  (static allocation only)
- - NO floating point
- - NO division  (bit operations only)
- - NO standard library except <stdint.h>
- - PROGMEM transparent via read callback
- - C99 compliant
    */

#ifndef BAD_DECODE_H
#define BAD_DECODE_H

/* ============================================================

- Platform detection
- ============================================================ */

#if defined(**AVR**)
/* ATmega328 / LGT8F328 ———————————— */
#include <stdint.h>
#define BAD_PLATFORM_AVR
#define BAD_PLATFORM_STR        “AVR”
#define BAD_INLINE              inline
#define BAD_STATIC_INLINE       static inline
/** AVR は uint16_t で十分（Flash 32KB上限） */
typedef uint16_t bad_addr_t;

#elif defined(ESP_PLATFORM) || defined(IDF_VER)
/* ESP32 (ESP-IDF) —————————————– */
#include <stdint.h>
#define BAD_PLATFORM_ESP32
#define BAD_PLATFORM_STR        “ESP32”
#define BAD_INLINE              inline **attribute**((always_inline))
#define BAD_STATIC_INLINE       static inline **attribute**((always_inline))
typedef uint32_t bad_addr_t;

#elif defined(ARDUINO_ARCH_RP2040) || defined(PICO_BOARD)   
|| defined(TARGET_RP2040)       || defined(TARGET_RP2350)
/* RP2040 / RP2350 (PlatformIO / Pico SDK) —————– */
#include <stdint.h>
#define BAD_PLATFORM_RP2
#define BAD_PLATFORM_STR        “RP2”
#define BAD_INLINE              inline **attribute**((always_inline))
#define BAD_STATIC_INLINE       static inline **attribute**((always_inline))
typedef uint32_t bad_addr_t;

#else
/* Unknown / Host (unit test) —————————— */
#include <stdint.h>
#define BAD_PLATFORM_UNKNOWN
#define BAD_PLATFORM_STR        “UNKNOWN”
#define BAD_INLINE              inline
#define BAD_STATIC_INLINE       static inline
typedef uint32_t bad_addr_t;
#endif

/* ============================================================

- NULL definition (no <stddef.h> dependency)
- ============================================================ */
  #ifndef NULL
  #define NULL ((void*)0)
  #endif

/* ============================================================

- Version / Protocol
- ============================================================ */

#define BAD_VERSION_STR          “0.5.1”
#define BAD_PROTOCOL_VERSION     ((uint16_t)510)

/* ============================================================

- File format constants
- ============================================================ */

/** ヘッダー全体サイズ (固定 19 バイト) */
#define BAD_HEADER_SIZE          ((uint8_t)19)

/** ヘッダー本体サイズ (固定 15 バイト) */
#define BAD_HEADER_BODY_SIZE     ((uint8_t)15)

/** ブロックサイズ (固定 8 px) */
#define BAD_BLOCK_SIZE           ((uint8_t)8)

/** ブロック内ピクセル数 */
#define BAD_BLOCK_PIXELS         ((uint8_t)64)

/** ブロック内バイト数 (64bit / 8) */
#define BAD_BLOCK_BYTES          ((uint8_t)8)

/* ============================================================

- Frame-level opcodes  (SPEC.md 4-2節)
- ============================================================ */

#define BAD_OP_BLOCK_STREAM      ((uint8_t)0x30)
#define BAD_OP_FRAME_DELIMITER   ((uint8_t)0x38)
#define BAD_OP_SKIP_FRAME        ((uint8_t)0x39)
#define BAD_OP_FILL_BLACK        ((uint8_t)0x3A)
#define BAD_OP_FILL_WHITE        ((uint8_t)0x3B)
#define BAD_OP_MASTER_BLOCK      ((uint8_t)0x3C)
#define BAD_OP_RLE_FRAME         ((uint8_t)0x3D)
#define BAD_OP_MASTER_FRAME      ((uint8_t)0x3E)
#define BAD_OP_INVERT_PREV       ((uint8_t)0x3F)
#define BAD_OP_EXT_PREFIX        ((uint8_t)0xFF)

/* ============================================================

- Block-level opcode classifiers  (SPEC.md 第5章)
- 
- 全てマクロで実装。関数呼び出しコストゼロ。
- AVR の 2 クロック分岐に最適化される。
- ============================================================ */

/** SKIP_BLOCK  10nnnnnn  0x80-0xBF */
#define BAD_IS_SKIP(op)          (((op) & 0xC0U) == 0x80U)
#define BAD_SKIP_COUNT(op)       (((op) & 0x3FU) + 1U)

/** FOR         11nnnnnn  0xC0-0xFF  繰り返し = n+2 */
#define BAD_IS_FOR(op)           (((op) & 0xC0U) == 0xC0U)
#define BAD_FOR_COUNT(op)        (((op) & 0x3FU) + 2U)

/** SHIFT_BIT   01sxxyy  0x40-0x7F */
#define BAD_IS_SHIFT(op)         (((op) & 0xC0U) == 0x40U)
#define BAD_SHIFT_SIGN_X(op)     (((op) >> 5U) & 0x01U)   /* 0=+ 1=- */
#define BAD_SHIFT_MAG_X(op)      (((op) >> 3U) & 0x03U)   /* 0-3 */
#define BAD_SHIFT_SIGN_Y(op)     (((op) >> 2U) & 0x01U)   /* 0=+ 1=- */
#define BAD_SHIFT_MAG_Y(op)      ( (op)         & 0x03U)   /* 0-3 */

/** BLOCK_INVERT  000nnnnn  0x00-0x1F */
#define BAD_IS_INVERT(op)        (((op) & 0xE0U) == 0x00U)
#define BAD_INVERT_COUNT(op)     (((op) & 0x1FU) + 1U)

/** RLE_BLOCK   0010pppc  0x20-0x2F */
#define BAD_IS_RLE(op)           (((op) & 0xF0U) == 0x20U)
#define BAD_RLE_PATTERN(op)      (((op) >> 1U) & 0x07U)   /* 走査パターン 0-7 */
#define BAD_RLE_START_COLOR(op)  ( (op)          & 0x01U)  /* 0=黒 1=白 */

/** FILL_BLOCK  00110cnn  0x30-0x37 */
#define BAD_IS_FILL(op)          (((op) & 0xF8U) == 0x30U)
#define BAD_FILL_COLOR(op)       (((op) >> 2U) & 0x01U)   /* 0=黒 1=白 */
#define BAD_FILL_COUNT(op)       (((op) & 0x03U) + 1U)    /* 1-4 */

/** MASTER_BLOCK  0x3C (FRAME_CONTROL 内) */
#define BAD_IS_MASTER_BLOCK(op)  ((op) == BAD_OP_MASTER_BLOCK)

/* ============================================================

- RLE_FRAME byte layout
- ============================================================ */

/** bit7=色, bit6-0=ラン長(1-127)  ラン長0は終端 */
#define BAD_RLEFRAME_COLOR(b)    (((b) >> 7U) & 0x01U)
#define BAD_RLEFRAME_LEN(b)      ( (b)         & 0x7FU)

/* ============================================================

- RLE_BLOCK bit-pack layout  (SPEC.md 6-5節)
- 6bit × 4 runs → 3 bytes リトルエンディアン
- 
- Byte0 = run0[5:0]          | run1[1:0] << 6
- Byte1 = run1[5:2] >> 2     | run2[3:0] << 4
- Byte2 = run2[5:4] >> 4     | run3[5:0] << 2
- ============================================================ */

#define BAD_RLE_UNPACK_R0(b0)         ( (b0)        & 0x3FU)
#define BAD_RLE_UNPACK_R1(b0, b1)     ((((b0) >> 6U) & 0x03U) | (((b1) & 0x0FU) << 2U))
#define BAD_RLE_UNPACK_R2(b1, b2)     ((((b1) >> 4U) & 0x0FU) | (((b2) & 0x03U) << 4U))
#define BAD_RLE_UNPACK_R3(b2)         (((b2) >> 2U) & 0x3FU)

/* ============================================================

- Result codes
- ============================================================ */

typedef enum {
BAD_OK          = 0,  /**< フレーム展開完了                         */
BAD_BUSY        = 1,  /**< 処理中（非同期拡張用・現行では未使用）    */
BAD_EOF         = 2,  /**< 全フレーム再生完了                       */
BAD_ERR_HDR     = 3,  /**< Fletcher-16 不一致 / ヘッダー構造不正    */
BAD_ERR_MAGIC   = 4,  /**< マジックナンバー “Bad” 不一致            */
BAD_ERR_DATA    = 5,  /**< フレームデータ破損 / 未知オペコード       */
BAD_ERR_MEM     = 6,  /**< gram/prev バッファサイズ不足             */
BAD_ERR_ARG     = 7,  /**< 引数不正 (NULL ポインタ等)               */
} bad_result_t;

/* ============================================================

- Data source callback
- 
- デコーダはこの関数を通じてのみデータにアクセスする。
- ターゲットごとにこの関数だけ実装すればよい。
- 
- AVR  : pgm_read_byte() / SD ライブラリ
- ESP32: SPIFFS / HTTPS stream / SPI Flash
- RP2  : LittleFS / DMA read / XIP Flash
- 
- @param offset  ファイル先頭からのバイトオフセット
- @param buf     読み出し先バッファ
- @param len     読み出し要求バイト数
- @return        実際に読み出せたバイト数
- ```
             (len 未満の場合は EOF 近傍)
  ```
- ============================================================ */

typedef uint16_t (*bad_read_fn)(bad_addr_t offset,
uint8_t   *buf,
uint16_t   len);

/* ============================================================

- Decoder context  (bad_ctx_t)
- 
- malloc は使用しない。
- 呼び出し側が static で確保する。
- 
- — 確保例 (AVR) —
- static uint8_t    gram[128*64/8];
- static uint8_t    prev[128*64/8];
- static bad_ctx_t  ctx;
- 
- — 使用手順 —
- 1. memset(&ctx, 0, sizeof(ctx))  または  = {0} で初期化
- 1. ctx.read     = my_read_func;
- ```
   ctx.gram     = gram;
  ```
- ```
   ctx.prev     = prev;
  ```
- ```
   ctx.buf_size = sizeof(gram);
  ```
- 1. bad_init(&ctx)  を呼ぶ
- 1. bad_next_frame(&ctx) をループで呼ぶ
- 1. BAD_EOF なら bad_rewind(&ctx) でループ再生
- ============================================================ */

typedef struct {

```
/* ---- 呼び出し側が設定するフィールド ---------------------- */

/** データ読み出しコールバック（必須） */
bad_read_fn  read;

/**
 * 現フレームバッファ（呼び出し側で確保・必須）
 * 必要サイズ: BAD_GRAM_SIZE(width, height)
 * bad_next_frame() 完了後にこのバッファをディスプレイへ転送する
 */
uint8_t     *gram;

/**
 * 前フレームバッファ（呼び出し側で確保・必須）
 * gram と同じサイズを確保すること
 * SKIP / SHIFT / INVERT 系命令で参照される
 * 呼び出し側から直接操作してはならない
 */
uint8_t     *prev;

/** gram / prev のバイト数（両者は同サイズであること） */
uint16_t     buf_size;

/* ---- デコーダが管理するフィールド（直接変更禁止） --------- */

uint16_t     width;          /**< 画像幅（ピクセル）               */
uint16_t     height;         /**< 画像高さ（ピクセル）             */
uint16_t     total_frames;   /**< ファイル内の総フレーム数         */
uint16_t     current_frame;  /**< 次に展開するフレーム番号(0始まり)*/
bad_addr_t   stream_offset;  /**< 次フレームの読み取り開始位置     */
uint8_t      initialized;    /**< bad_init() 完了フラグ (0/1)     */
```

} bad_ctx_t;

/* ============================================================

- Public API
- ============================================================ */

#ifdef __cplusplus
extern “C” {
#endif

/**

- @brief デコーダを初期化してヘッダーを検証する
- 
- 事前に ctx->read / gram / prev / buf_size を設定すること。
- 成功時: ctx->width / height / total_frames が設定される。
- 
- @param ctx  デコーダコンテキスト
- @return     BAD_OK / BAD_ERR_ARG / BAD_ERR_HDR /
- ```
          BAD_ERR_MAGIC / BAD_ERR_MEM
  ```

*/
bad_result_t bad_init(bad_ctx_t *ctx);

/**

- @brief 次のフレームを gram[] に展開する
- 
- 内部処理順:
- 1. prev[] ← gram[] コピー（前フレーム保存）
- 1. ストリームから次フレームを読み取り gram[] に展開
- 1. ctx->current_frame インクリメント
- 1. BAD_OK または BAD_EOF を返す
- 
- 呼び出し後 ctx->gram をディスプレイに転送すること。
- 
- @param ctx  bad_init() 済みのコンテキスト
- @return     BAD_OK / BAD_EOF / BAD_ERR_DATA / BAD_ERR_ARG
  */
  bad_result_t bad_next_frame(bad_ctx_t *ctx);

/**

- @brief 先頭フレームに巻き戻す
- 
- current_frame = 0 / stream_offset をヘッダー直後に戻す。
- prev[] をゼロクリアする。
- 
- @param ctx  デコーダコンテキスト
- @return     BAD_OK / BAD_ERR_ARG
  */
  bad_result_t bad_rewind(bad_ctx_t *ctx);

/**

- @brief 任意フレームへシークする
- 
- @warning ストリームを先頭から走査するため低速。
- ```
       マイコン用途では bad_rewind() によるループ再生を推奨。
  ```
- 
- @param ctx       デコーダコンテキスト
- @param frame_no  移動先フレーム番号（0始まり）
- @return          BAD_OK / BAD_EOF / BAD_ERR_DATA / BAD_ERR_ARG
  */
  bad_result_t bad_seek(bad_ctx_t *ctx, uint16_t frame_no);

/**

- @brief 現在のフレーム番号を返す（0始まり）
  */
  BAD_STATIC_INLINE uint16_t bad_current_frame(const bad_ctx_t *ctx)
  {
  return (ctx != NULL) ? ctx->current_frame : 0U;
  }

/**

- @brief 総フレーム数を返す（bad_init() 後に有効）
  */
  BAD_STATIC_INLINE uint16_t bad_total_frames(const bad_ctx_t *ctx)
  {
  return (ctx != NULL) ? ctx->total_frames : 0U;
  }

/**

- @brief エラーコードを文字列に変換する（デバッグ用）
- 
- @note Flash を節約したい場合はこの関数をリンクしないこと。
- ```
    エラーコードを数値のまま扱えば Flash 消費はゼロ。
  ```

*/
const char *bad_result_str(bad_result_t result);

#ifdef __cplusplus
}
#endif

/* ============================================================

- Utility macros
- ============================================================ */

/**

- gram / prev バッファの必要バイト数を計算する
- @param w  画像幅（ピクセル）
- @param h  画像高さ（ピクセル）
  */
  #define BAD_GRAM_SIZE(w, h)      ((uint16_t)(((uint16_t)(w) * (uint16_t)(h)) >> 3U))

/**

- gram[] から指定ピクセルのビットを取得する
- リトルエンディアン・行方向（左→右, 上→下）
- @return  0=黒  1=白
  */
  #define BAD_GET_PIXEL(gram, x, y, w)   
  (((gram)[((uint16_t)(y) * (uint16_t)(w) + (uint16_t)(x)) >> 3U] \

> > 
> > (((uint16_t)(y) * (uint16_t)(w) + (uint16_t)(x)) & 0x07U)) & 0x01U)

/* ============================================================

- Usage example (Arduino / AVR)
- 
- #include “bad_decode.h”
- 
- static uint8_t   gram[BAD_GRAM_SIZE(128, 64)];
- static uint8_t   prev[BAD_GRAM_SIZE(128, 64)];
- static bad_ctx_t ctx;
- 
- uint16_t my_read(bad_addr_t offset, uint8_t *buf, uint16_t len) {
- ```
    // Flash / SD / SPI etc.
  ```
- ```
    for (uint16_t i = 0; i < len; i++)
  ```
- ```
        buf[i] = pgm_read_byte(VIDEO_DATA + offset + i);
  ```
- ```
    return len;
  ```
- }
- 
- void setup() {
- ```
    ctx.read     = my_read;
  ```
- ```
    ctx.gram     = gram;
  ```
- ```
    ctx.prev     = prev;
  ```
- ```
    ctx.buf_size = sizeof(gram);
  ```
- ```
    bad_init(&ctx);
  ```
- }
- 
- void loop() {
- ```
    bad_result_t r = bad_next_frame(&ctx);
  ```
- ```
    if (r == BAD_OK)  display_write(ctx.gram);
  ```
- ```
    if (r == BAD_EOF) bad_rewind(&ctx);
  ```
- }
- ============================================================ */

#endif /* BAD_DECODE_H */