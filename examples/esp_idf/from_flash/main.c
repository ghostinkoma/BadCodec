/**
 * @file  main.c
 * @brief BadCodec サンプル: ESP-IDF / ヘッダファイル (bad_data.h) から再生
 * @version v0.6.0  (Protocol 514, SPEC rev.19)
 *
 * ビルド方法 (ESP-IDF 5.x):
 *   idf.py build flash monitor
 *
 * 必要ファイル:
 *   bad_decode.h / bad_decode.cpp  → components/bad_codec/
 *   bad_data.h                     → main/
 *
 * ディスプレイ: SSD1306 128x64 (esp-idf-lib / ssd1306 コンポーネント使用)
 * 接続: SDA=GPIO21  SCL=GPIO22
 */

#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "bad_decode.h"
#include "bad_data.h"      /* Codec.py -t c で生成 */

/* ============================================================
 * ディスプレイドライバ (ユーザー実装)
 * esp-idf-lib の ssd1306 コンポーネントを使用する例
 * ============================================================ */
#include "ssd1306.h"   /* https://github.com/UncleRus/esp-idf-lib */

#define I2C_PORT   I2C_NUM_0
#define SDA_GPIO   21
#define SCL_GPIO   22
#define SSD1306_ADDR 0x3C

static ssd1306_t disp;

static void display_init(void)
{
    i2c_config_t cfg = {
        .mode             = I2C_MODE_MASTER,
        .sda_io_num       = SDA_GPIO,
        .scl_io_num       = SCL_GPIO,
        .sda_pullup_en    = GPIO_PULLUP_ENABLE,
        .scl_pullup_en    = GPIO_PULLUP_ENABLE,
        .master.clk_speed = 400000,
    };
    i2c_param_config(I2C_PORT, &cfg);
    i2c_driver_install(I2C_PORT, cfg.mode, 0, 0, 0);

    disp.i2c_port = I2C_PORT;
    disp.addr     = SSD1306_ADDR;
    disp.width    = 128;
    disp.height   = 64;
    ssd1306_init(&disp);
    ssd1306_clear(&disp);
}

/*
 * gram は 1bit/pixel, LSB first, row-major
 * SSD1306 は列方向に8px=1byte の形式が多い。
 * ここではシンプルな変換例を示す。
 * 実際のドライバに合わせて変更すること。
 */
static void display_frame(const uint8_t *gram, uint16_t w, uint16_t h)
{
    /* SSD1306 の GDDRAM 形式 (page mode): 列単位で変換 */
    static uint8_t page_buf[128 * 8];  /* 128x64 / 8 = 1024 bytes */
    memset(page_buf, 0, sizeof(page_buf));

    for (uint16_t y = 0; y < h; y++) {
        for (uint16_t x = 0; x < w; x++) {
            uint16_t bit_idx = y * w + x;
            uint8_t  pix     = (gram[bit_idx >> 3] >> (bit_idx & 7)) & 1;
            if (pix) {
                uint16_t page = y >> 3;
                uint8_t  bit  = y & 7;
                page_buf[page * w + x] |= (uint8_t)(1 << bit);
            }
        }
    }
    ssd1306_draw_bitmap(&disp, 0, 0, page_buf, w, h);
    ssd1306_update(&disp);
}

/* ============================================================
 * BadCodec デコーダ
 * ============================================================ */
static uint8_t   gram[BAD_GRAM_SIZE(BAD_DATA_WIDTH, BAD_DATA_HEIGHT)];
static uint8_t   prev_buf[BAD_GRAM_SIZE(BAD_DATA_WIDTH, BAD_DATA_HEIGHT)];
static bad_ctx_t ctx;

/* read コールバック: ROM (Flash キャッシュ) から直接読む */
static uint16_t flash_read(bad_addr_t offset, uint8_t *buf, uint16_t len)
{
    memcpy(buf, bad_data + offset, len);
    return len;
}

/* ============================================================
 * メインタスク
 * ============================================================ */
static void video_task(void *arg)
{
    display_init();

    ctx.read     = flash_read;
    ctx.gram     = gram;
    ctx.prev     = prev_buf;
    ctx.buf_size = sizeof(gram);

    if (bad_init(&ctx) != BAD_OK) {
        ESP_LOGE("BadCodec", "init failed");
        vTaskDelete(NULL);
    }

    ESP_LOGI("BadCodec", "Playing %ux%u %u frames",
             ctx.width, ctx.height, ctx.total_frames);

    for (;;) {
        TickType_t t0 = xTaskGetTickCount();

        bad_result_t r = bad_next_frame(&ctx);
        if (r == BAD_OK || r == BAD_EOF) {
            display_frame(ctx.gram, ctx.width, ctx.height);
        }
        if (r == BAD_EOF) {
            bad_rewind(&ctx);
        }

        /* 約30fps */
        vTaskDelayUntil(&t0, pdMS_TO_TICKS(33));
    }
}

void app_main(void)
{
    xTaskCreate(video_task, "video", 4096, NULL, 5, NULL);
}
