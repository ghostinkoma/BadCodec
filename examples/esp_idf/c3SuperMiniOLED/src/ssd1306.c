/**
 * @file  ssd1306.c
 * @brief SSD1306 128x64 OLED ドライバ実装
 * @version v0.6.0
 *
 * 参考コード (main.c) の I2C / OLED 部分をそのまま移植。
 * GPIO・アドレス・周波数は bad_player_config.h で一元管理。
 */

#include "ssd1306.h"
#include "driver/i2c.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <string.h>

/* ============================================================
 * framebuffer (外部公開)
 * ============================================================ */
uint8_t ssd1306_fb[BAD_OLED_WIDTH * BAD_OLED_PAGES];

/* ============================================================
 * 内部: I2C コマンド / データ送信
 * ============================================================ */
static void oled_cmd(uint8_t cmd)
{
    i2c_cmd_handle_t h = i2c_cmd_link_create();
    i2c_master_start(h);
    i2c_master_write_byte(h, (BAD_OLED_ADDR << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(h, 0x00, true);   /* Co=0, D/C#=0 → command */
    i2c_master_write_byte(h, cmd, true);
    i2c_master_stop(h);
    i2c_master_cmd_begin(BAD_I2C_PORT, h, pdMS_TO_TICKS(100));
    i2c_cmd_link_delete(h);
}

static void oled_data(const uint8_t *data, size_t len)
{
    i2c_cmd_handle_t h = i2c_cmd_link_create();
    i2c_master_start(h);
    i2c_master_write_byte(h, (BAD_OLED_ADDR << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(h, 0x40, true);   /* Co=0, D/C#=1 → data */
    i2c_master_write(h, data, len, true);
    i2c_master_stop(h);
    i2c_master_cmd_begin(BAD_I2C_PORT, h, pdMS_TO_TICKS(100));
    i2c_cmd_link_delete(h);
}

/* ============================================================
 * ssd1306_init
 * ============================================================ */
void ssd1306_init(void)
{
    /* I2C マスタ初期化 */
    i2c_config_t conf = {
        .mode             = I2C_MODE_MASTER,
        .sda_io_num       = BAD_I2C_SDA,
        .scl_io_num       = BAD_I2C_SCL,
        .sda_pullup_en    = GPIO_PULLUP_ENABLE,
        .scl_pullup_en    = GPIO_PULLUP_ENABLE,
        .master.clk_speed = BAD_I2C_FREQ_HZ,
    };
    i2c_param_config(BAD_I2C_PORT, &conf);
    i2c_driver_install(BAD_I2C_PORT, conf.mode, 0, 0, 0);

    /* SSD1306 初期化シーケンス (参考コードと同一) */
    oled_cmd(0xAE);         /* Display OFF */
    oled_cmd(0xA8); oled_cmd(0x3F); /* MUX ratio = 64 */
    oled_cmd(0xD3); oled_cmd(0x00); /* Display offset = 0 */
    oled_cmd(0x40);         /* Display start line = 0 */
    oled_cmd(0xA1);         /* Segment remap: col127 → SEG0 */
    oled_cmd(0xC8);         /* COM scan direction: remapped */
    oled_cmd(0xDA); oled_cmd(0x12); /* COM pins config */
    oled_cmd(0x81); oled_cmd(0x7F); /* Contrast = 127 */
    oled_cmd(0xA4);         /* Entire display ON (follow RAM) */
    oled_cmd(0xA6);         /* Normal display (not inverted) */
    oled_cmd(0xD5); oled_cmd(0x80); /* Display clock divide */
    oled_cmd(0x20); oled_cmd(0x02); /* Memory mode: page addressing */
    oled_cmd(0x8D); oled_cmd(0x14); /* Charge pump ON */
    oled_cmd(0xAF);         /* Display ON */

    ssd1306_clear();
    ssd1306_flush();
}

/* ============================================================
 * ssd1306_clear
 * ============================================================ */
void ssd1306_clear(void)
{
    memset(ssd1306_fb, 0x00, sizeof(ssd1306_fb));
}

/* ============================================================
 * ssd1306_flush
 * 参考コードの oled_flush_offset と同一ロジック
 * ============================================================ */
void ssd1306_flush(void)
{
    for (int p = 0; p < BAD_OLED_PAGES; p++) {
        oled_cmd(0xB0 | p);    /* page address */
        oled_cmd(0x00);         /* lower column = 0 */
        oled_cmd(0x10);         /* upper column = 0 */
        oled_data(&ssd1306_fb[p * BAD_OLED_WIDTH], BAD_OLED_WIDTH);
    }
}

/* ============================================================
 * ssd1306_draw_gram
 *
 * BadCodec gram (1bit/px, LSB first, row-major) を
 * SSD1306 framebuffer (page 形式) に変換して貼り付ける。
 *
 * gram のビット配置:
 *   pixel(x,y) = (gram[y * w/8 + x/8] >> (x%8)) & 1
 *
 * fb のビット配置:
 *   fb[page * 128 + phys_x]  bit(py%8) = pixel(phys_x, py)
 *   page = py / 8,  py = y + y_off,  phys_x = x + x_off
 * ============================================================ */
void ssd1306_draw_gram(const uint8_t *gram,
                       uint16_t w, uint16_t h,
                       int x_off, int y_off)
{
    for (int y = 0; y < (int)h; y++) {
        int py = y + y_off;
        if (py < 0 || py >= BAD_OLED_HEIGHT) continue;
        int page = py >> 3;          /* py / 8 */
        int bit  = py &  7;          /* py % 8 */

        for (int x = 0; x < (int)w; x++) {
            int px = x + x_off;
            if (px < 0 || px >= BAD_OLED_WIDTH) continue;

            /* gram から 1bit 取得 */
            uint16_t bi  = (uint16_t)(y * w + x);
            uint8_t  pix = (gram[bi >> 3] >> (bi & 7)) & 1U;

            /* fb に書き込み */
            uint16_t fi = (uint16_t)(page * BAD_OLED_WIDTH + px);
            if (pix)
                ssd1306_fb[fi] |=  (uint8_t)(1U << bit);
            else
                ssd1306_fb[fi] &= ~(uint8_t)(1U << bit);
        }
    }
}
