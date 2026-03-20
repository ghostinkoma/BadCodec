/**
 * @file  from_flash.ino
 * @brief BadCodec サンプル: ヘッダファイル (bad_data.h) から再生
 *
 * ボード: LGT8F328P / ATmega328P
 * ディスプレイ: SSD1306 128x64 OLED (U8g2 ライブラリ使用)
 *
 * 事前準備:
 *   1. Codec.py で .bad ファイルをエンコード
 *   2. -t c -i output.bad -H bad_data.h でヘッダを生成
 *   3. bad_data.h をこのスケッチと同じフォルダに配置
 *   4. ライブラリ: U8g2 (Library Manager からインストール)
 *
 * 接続:
 *   SSD1306 SDA → A4 (Arduino Uno)
 *   SSD1306 SCL → A5 (Arduino Uno)
 */

#include <Arduino.h>
#include <U8g2lib.h>
#include <Wire.h>
#include "bad_decode.h"
#include "bad_data.h"      /* Codec.py -t c で生成したヘッダ */

/* ============================================================
 * ディスプレイ設定 (使用するディスプレイに合わせて変更)
 * ============================================================ */
U8G2_SSD1306_128X64_NONAME_1_HW_I2C u8g2(U8G2_R0, U8X8_PIN_NONE);

/* ============================================================
 * BadCodec デコーダ
 * ============================================================ */
static uint8_t   gram[BAD_GRAM_SIZE(BAD_DATA_WIDTH, BAD_DATA_HEIGHT)];
static uint8_t   prev[BAD_GRAM_SIZE(BAD_DATA_WIDTH, BAD_DATA_HEIGHT)];
static bad_ctx_t ctx;

/* read コールバック: PROGMEM (Flash) から読む */
static uint16_t flash_read(bad_addr_t offset, uint8_t *buf, uint16_t len)
{
    for (uint16_t i = 0; i < len; i++)
        buf[i] = pgm_read_byte(bad_data + offset + i);
    return len;
}

/* ============================================================
 * 表示関数
 * gram は 1bit/pixel, LSB first, row-major
 * U8g2 の XBM 形式と同一なのでそのまま渡せる
 * ============================================================ */
static void display_frame(const uint8_t *g, uint16_t w, uint16_t h)
{
    u8g2.firstPage();
    do {
        u8g2.drawXBM(0, 0, w, h, g);
    } while (u8g2.nextPage());
}

/* ============================================================
 * Setup / Loop
 * ============================================================ */
void setup()
{
    Wire.begin();
    u8g2.begin();
    u8g2.setContrast(128);

    ctx.read      = flash_read;
    ctx.gram      = gram;
    ctx.prev      = prev;
    ctx.buf_size  = sizeof(gram);

    bad_result_t r = bad_init(&ctx);
    if (r != BAD_OK) {
        u8g2.clearBuffer();
        u8g2.setFont(u8g2_font_6x10_tf);
        u8g2.drawStr(0, 12, "BadCodec init failed");
        u8g2.sendBuffer();
        for (;;);
    }
}

void loop()
{
    uint32_t t0 = millis();

    bad_result_t r = bad_next_frame(&ctx);
    if (r == BAD_OK || r == BAD_EOF) {
        display_frame(ctx.gram, ctx.width, ctx.height);
    }
    if (r == BAD_EOF) {
        bad_rewind(&ctx);
    }

    /* 約30fps (33ms/frame) に調整 */
    uint32_t elapsed = millis() - t0;
    if (elapsed < 33)
        delay(33 - elapsed);
}
