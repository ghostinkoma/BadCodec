/**
 * @file  main.c
 * @brief BadCodec サンプル: RP2040 / RP2350 / ヘッダファイル (bad_data.h) から再生
 *
 * ビルド方法 (Pico SDK 2.x):
 *   mkdir build && cd build
 *   cmake .. -DPICO_BOARD=pico2
 *   make -j4
 *
 * ディスプレイ: SSD1306 128x64 OLED (pico-ssd1306 ライブラリ)
 * 接続: SDA=GP4  SCL=GP5
 *
 * 必要ファイル:
 *   bad_decode.h / bad_decode.cpp  (プロジェクトルートに配置)
 *   bad_data.h                     (Codec.py -t c で生成)
 */

#include <string.h>
#include "pico/stdlib.h"
#include "pico/time.h"
#include "hardware/i2c.h"
#include "bad_decode.h"
#include "bad_data.h"    /* Codec.py -t c で生成したヘッダ */

/* ============================================================
 * SSD1306 最小ドライバ (pico-ssd1306 がない場合の代替)
 * ============================================================ */
#define SSD1306_I2C     i2c0
#define SSD1306_SDA     4
#define SSD1306_SCL     5
#define SSD1306_ADDR    0x3C
#define SSD1306_W       128
#define SSD1306_H       64

static void ssd1306_cmd(uint8_t cmd)
{
    uint8_t buf[2] = {0x00, cmd};
    i2c_write_blocking(SSD1306_I2C, SSD1306_ADDR, buf, 2, false);
}

static void ssd1306_init_seq(void)
{
    sleep_ms(100);
    static const uint8_t cmds[] = {
        0xAE, 0x20, 0x00, 0x40, 0xA1, 0xC8, 0xA8, 0x3F,
        0xD3, 0x00, 0xDA, 0x12, 0xD5, 0x80, 0xD9, 0xF1,
        0xDB, 0x30, 0x81, 0xFF, 0xA4, 0xA6, 0x8D, 0x14, 0xAF
    };
    for (size_t i = 0; i < sizeof(cmds); i++)
        ssd1306_cmd(cmds[i]);
}

/*
 * gram (1bit row-major LSB first) → SSD1306 GDDRAM (page-mode) に変換して送信
 */
static void ssd1306_flush(const uint8_t *gram, uint16_t w, uint16_t h)
{
    static uint8_t page_buf[SSD1306_W * (SSD1306_H / 8)];
    memset(page_buf, 0, sizeof(page_buf));

    for (uint16_t y = 0; y < h; y++) {
        for (uint16_t x = 0; x < w; x++) {
            uint16_t bi  = y * w + x;
            uint8_t  pix = (gram[bi >> 3] >> (bi & 7)) & 1;
            if (pix) {
                page_buf[(y >> 3) * w + x] |= (uint8_t)(1U << (y & 7));
            }
        }
    }

    /* 全ページ一括送信 */
    ssd1306_cmd(0x21); ssd1306_cmd(0); ssd1306_cmd(127);
    ssd1306_cmd(0x22); ssd1306_cmd(0); ssd1306_cmd(7);
    uint8_t header = 0x40;
    i2c_write_blocking(SSD1306_I2C, SSD1306_ADDR, &header, 1, true);
    i2c_write_blocking(SSD1306_I2C, SSD1306_ADDR,
                       page_buf, sizeof(page_buf), false);
}

/* ============================================================
 * BadCodec デコーダ
 * ============================================================ */
static uint8_t   gram_buf[BAD_GRAM_SIZE(BAD_DATA_WIDTH, BAD_DATA_HEIGHT)];
static uint8_t   prev_buf[BAD_GRAM_SIZE(BAD_DATA_WIDTH, BAD_DATA_HEIGHT)];
static bad_ctx_t ctx;

/* read コールバック: Flash (XIP) からコピー */
static uint16_t flash_read(bad_addr_t offset, uint8_t *buf, uint16_t len)
{
    memcpy(buf, bad_data + offset, len);
    return len;
}

/* ============================================================
 * main
 * ============================================================ */
int main(void)
{
    stdio_init_all();

    /* I2C 初期化 */
    i2c_init(SSD1306_I2C, 400 * 1000);
    gpio_set_function(SSD1306_SDA, GPIO_FUNC_I2C);
    gpio_set_function(SSD1306_SCL, GPIO_FUNC_I2C);
    gpio_pull_up(SSD1306_SDA);
    gpio_pull_up(SSD1306_SCL);

    ssd1306_init_seq();

    /* BadCodec 初期化 */
    ctx.read     = flash_read;
    ctx.gram     = gram_buf;
    ctx.prev     = prev_buf;
    ctx.buf_size = sizeof(gram_buf);

    if (bad_init(&ctx) != BAD_OK) {
        /* エラー: 画面に "E" を表示 */
        memset(gram_buf, 0, sizeof(gram_buf));
        gram_buf[0] = 0xFF;
        ssd1306_flush(gram_buf, SSD1306_W, SSD1306_H);
        for (;;) tight_loop_contents();
    }

    /* 再生ループ (約30fps) */
    for (;;) {
        absolute_time_t t0 = get_absolute_time();

        bad_result_t r = bad_next_frame(&ctx);
        if (r == BAD_OK || r == BAD_EOF) {
            ssd1306_flush(ctx.gram, ctx.width, ctx.height);
        }
        if (r == BAD_EOF) {
            bad_rewind(&ctx);
        }

        int64_t elapsed_us = absolute_time_diff_us(t0, get_absolute_time());
        int64_t wait_us    = 33333 - elapsed_us;  /* 33.333ms = 30fps */
        if (wait_us > 0)
            sleep_us((uint32_t)wait_us);
    }

    return 0;
}
