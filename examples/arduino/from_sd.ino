/**
 * @file  from_sd.ino
 * @brief BadCodec サンプル: SD カードから再生
 *
 * ボード: LGT8F328P / ATmega328P
 * ディスプレイ: SSD1306 128x64 OLED (U8g2)
 * SD カード: SPI 接続
 *
 * 接続:
 *   SSD1306 SDA → A4  SSD1306 SCL → A5
 *   SD CS  → D10  SD MOSI → D11  SD MISO → D12  SD SCK → D13
 *
 * SD カードに output.bad を置いてください。
 */

#include <Arduino.h>
#include <U8g2lib.h>
#include <Wire.h>
#include <SD.h>
#include "bad_decode.h"

#define SD_CS_PIN    10
#define VIDEO_FILE   "output.bad"

/* ============================================================
 * ディスプレイ
 * ============================================================ */
U8G2_SSD1306_128X64_NONAME_1_HW_I2C u8g2(U8G2_R0, U8X8_PIN_NONE);

/* ============================================================
 * BadCodec デコーダ
 * ============================================================ */
static uint8_t   gram[BAD_GRAM_SIZE(128, 64)];
static uint8_t   prev[BAD_GRAM_SIZE(128, 64)];
static bad_ctx_t ctx;
static File      sd_file;

/* read コールバック: SD カードからシーケンシャル読み出し
 * offset が前回と同じ場合はシークしない (SD ライブラリの特性) */
static uint32_t  last_offset = 0xFFFFFFFFUL;

static uint16_t sd_read(bad_addr_t offset, uint8_t *buf, uint16_t len)
{
    if ((uint32_t)offset != last_offset) {
        sd_file.seek((uint32_t)offset);
    }
    uint16_t n = (uint16_t)sd_file.read(buf, len);
    last_offset = (uint32_t)offset + n;
    return n;
}

/* ============================================================
 * 表示
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

    /* SD 初期化 */
    if (!SD.begin(SD_CS_PIN)) {
        u8g2.clearBuffer();
        u8g2.setFont(u8g2_font_6x10_tf);
        u8g2.drawStr(0, 12, "SD init failed");
        u8g2.sendBuffer();
        for (;;);
    }

    sd_file = SD.open(VIDEO_FILE);
    if (!sd_file) {
        u8g2.clearBuffer();
        u8g2.setFont(u8g2_font_6x10_tf);
        u8g2.drawStr(0, 12, "File not found");
        u8g2.sendBuffer();
        for (;;);
    }

    ctx.read      = sd_read;
    ctx.gram      = gram;
    ctx.prev      = prev;
    ctx.buf_size  = sizeof(gram);

    if (bad_init(&ctx) != BAD_OK) {
        u8g2.clearBuffer();
        u8g2.setFont(u8g2_font_6x10_tf);
        u8g2.drawStr(0, 12, "BadCodec init err");
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
        /* ループ再生: ファイルの先頭にシーク */
        bad_rewind(&ctx);
        last_offset = 0xFFFFFFFFUL;
    }

    uint32_t elapsed = millis() - t0;
    if (elapsed < 33)
        delay(33 - elapsed);
}
