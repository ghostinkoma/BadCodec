/**
 * @file  main.c
 * @brief BadCodec サンプル: RP2040 / RP2350 / SD カードから再生
 *
 * SD カード: SPI 接続 (no_os_fatfs または FatFs ライブラリ)
 * ディスプレイ: SSD1306 128x64
 *
 * 接続:
 *   SD  CS=GP5  SCK=GP2  MOSI=GP3  MISO=GP4
 *   SSD1306 SDA=GP6  SCL=GP7
 *
 * SD カードのルートに output.bad を置いてください。
 *
 * FatFs ライブラリ:
 *   https://github.com/carlk3/no-OS-FatFS-SD-SDIO-SPI-RPi-Pico
 */

#include <stdio.h>
#include <string.h>
#include "pico/stdlib.h"
#include "pico/time.h"
#include "hardware/i2c.h"
#include "hardware/spi.h"
#include "ff.h"          /* FatFs */
#include "bad_decode.h"

#define VIDEO_FILE  "output.bad"

/* ============================================================
 * SSD1306 (from_flash と同一 - 省略)
 * ============================================================ */
#define SSD1306_I2C   i2c1
#define SSD1306_SDA   6
#define SSD1306_SCL   7
#define SSD1306_ADDR  0x3C
#define SSD1306_W     128
#define SSD1306_H     64

static void ssd1306_cmd(uint8_t c) {
    uint8_t b[2] = {0x00, c};
    i2c_write_blocking(SSD1306_I2C, SSD1306_ADDR, b, 2, false);
}
static void ssd1306_init_seq(void) {
    sleep_ms(100);
    const uint8_t cmds[] = {
        0xAE,0x20,0x00,0x40,0xA1,0xC8,0xA8,0x3F,
        0xD3,0x00,0xDA,0x12,0xD5,0x80,0xD9,0xF1,
        0xDB,0x30,0x81,0xFF,0xA4,0xA6,0x8D,0x14,0xAF
    };
    for (size_t i=0;i<sizeof(cmds);i++) ssd1306_cmd(cmds[i]);
}
static void ssd1306_flush(const uint8_t *gram, uint16_t w, uint16_t h) {
    static uint8_t pb[128*8];
    memset(pb,0,sizeof(pb));
    for(uint16_t y=0;y<h;y++) for(uint16_t x=0;x<w;x++) {
        uint16_t bi=(uint16_t)(y*w+x);
        if((gram[bi>>3]>>(bi&7))&1) pb[(y>>3)*w+x]|=(uint8_t)(1<<(y&7));
    }
    ssd1306_cmd(0x21);ssd1306_cmd(0);ssd1306_cmd(127);
    ssd1306_cmd(0x22);ssd1306_cmd(0);ssd1306_cmd(7);
    uint8_t hdr=0x40;
    i2c_write_blocking(SSD1306_I2C,SSD1306_ADDR,&hdr,1,true);
    i2c_write_blocking(SSD1306_I2C,SSD1306_ADDR,pb,sizeof(pb),false);
}

/* ============================================================
 * BadCodec デコーダ
 * ============================================================ */
static uint8_t   gram_buf[BAD_GRAM_SIZE(128, 64)];
static uint8_t   prev_buf[BAD_GRAM_SIZE(128, 64)];
static bad_ctx_t ctx;

static FIL    sd_file;
static FSIZE_t last_pos = (FSIZE_t)-1;

/* read コールバック: FatFs f_read */
static uint16_t sd_read(bad_addr_t offset, uint8_t *buf, uint16_t len)
{
    if ((FSIZE_t)offset != last_pos) {
        f_lseek(&sd_file, (FSIZE_t)offset);
    }
    UINT n = 0;
    f_read(&sd_file, buf, len, &n);
    last_pos = (FSIZE_t)offset + n;
    return (uint16_t)n;
}

/* ============================================================
 * main
 * ============================================================ */
int main(void)
{
    stdio_init_all();

    /* I2C 初期化 (SSD1306) */
    i2c_init(SSD1306_I2C, 400000);
    gpio_set_function(SSD1306_SDA, GPIO_FUNC_I2C);
    gpio_set_function(SSD1306_SCL, GPIO_FUNC_I2C);
    gpio_pull_up(SSD1306_SDA);
    gpio_pull_up(SSD1306_SCL);
    ssd1306_init_seq();

    /* SD / FatFs 初期化 */
    FATFS fs;
    if (f_mount(&fs, "", 1) != FR_OK) {
        printf("SD mount failed\n");
        for (;;) tight_loop_contents();
    }
    if (f_open(&sd_file, VIDEO_FILE, FA_READ) != FR_OK) {
        printf("Cannot open %s\n", VIDEO_FILE);
        for (;;) tight_loop_contents();
    }

    /* BadCodec 初期化 */
    ctx.read     = sd_read;
    ctx.gram     = gram_buf;
    ctx.prev     = prev_buf;
    ctx.buf_size = sizeof(gram_buf);

    if (bad_init(&ctx) != BAD_OK) {
        printf("bad_init failed\n");
        for (;;) tight_loop_contents();
    }
    printf("BadCodec: %ux%u %u frames\n",
           ctx.width, ctx.height, ctx.total_frames);

    /* 再生ループ */
    for (;;) {
        absolute_time_t t0 = get_absolute_time();

        bad_result_t r = bad_next_frame(&ctx);
        if (r == BAD_OK || r == BAD_EOF) {
            ssd1306_flush(ctx.gram, ctx.width, ctx.height);
        }
        if (r == BAD_EOF) {
            bad_rewind(&ctx);
            last_pos = (FSIZE_t)-1;
        }

        int64_t w_us = 33333 - absolute_time_diff_us(t0, get_absolute_time());
        if (w_us > 0) sleep_us((uint32_t)w_us);
    }

    return 0;
}
