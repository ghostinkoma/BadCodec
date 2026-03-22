/**
 * @file  ssd1306_drv.c
 * @brief SSD1306 ドライバ実装
 * @version v0.6.0
 *
 * 参考コード (旧 main.c) の I2C / OLED 部分を忠実に移植。
 * 変更点:
 *   - GPIO・アドレス等を config.h に外出し
 *   - draw_gram() を追加 (BadCodec gram → page-mode fb 変換)
 */

#include "ssd1306_drv.h"
#include "driver/i2c.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <string.h>

/* ============================================================
 * framebuffer
 * ============================================================ */
uint8_t g_fb[CFG_PHYS_W * CFG_PAGES];

/* ============================================================
 * 内部ヘルパー
 * ============================================================ */
static void send_cmd(uint8_t cmd)
{
    i2c_cmd_handle_t h = i2c_cmd_link_create();
    i2c_master_start(h);
    i2c_master_write_byte(h, (CFG_OLED_ADDR << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(h, 0x00, true);  /* D/C# = 0: command */
    i2c_master_write_byte(h, cmd,  true);
    i2c_master_stop(h);
    i2c_master_cmd_begin(CFG_I2C_PORT, h, pdMS_TO_TICKS(100));
    i2c_cmd_link_delete(h);
}

static void send_data(const uint8_t *data, size_t len)
{
    i2c_cmd_handle_t h = i2c_cmd_link_create();
    i2c_master_start(h);
    i2c_master_write_byte(h, (CFG_OLED_ADDR << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(h, 0x40, true);  /* D/C# = 1: data */
    i2c_master_write(h, data, len, true);
    i2c_master_stop(h);
    i2c_master_cmd_begin(CFG_I2C_PORT, h, pdMS_TO_TICKS(100));
    i2c_cmd_link_delete(h);
}

/* ============================================================
 * ssd1306_init
 * 参考コードの i2c_init() + oled_init() を統合
 * ============================================================ */
void ssd1306_init(void)
{
    /* I2C マスタ初期化 */
    i2c_config_t conf = {
        .mode             = I2C_MODE_MASTER,
        .sda_io_num       = CFG_I2C_SDA,
        .scl_io_num       = CFG_I2C_SCL,
        .sda_pullup_en    = GPIO_PULLUP_ENABLE,
        .scl_pullup_en    = GPIO_PULLUP_ENABLE,
        .master.clk_speed = CFG_I2C_FREQ_HZ,
    };
    i2c_param_config(CFG_I2C_PORT, &conf);
    i2c_driver_install(CFG_I2C_PORT, conf.mode, 0, 0, 0);

    /* SSD1306 初期化コマンド列 (参考コードと同一) */
    send_cmd(0xAE);              /* Display OFF */
    send_cmd(0xA8); send_cmd(0x3F); /* MUX ratio 64 */
    send_cmd(0xD3); send_cmd(0x00); /* Display offset 0 */
    send_cmd(0x40);              /* Start line 0 */
    send_cmd(0xA1);              /* SEG remap (col127=SEG0) */
    send_cmd(0xC8);              /* COM scan reverse */
    send_cmd(0xDA); send_cmd(0x12); /* COM pins */
    send_cmd(0x81); send_cmd(0x7F); /* Contrast 127 */
    send_cmd(0xA4);              /* Follow RAM */
    send_cmd(0xA6);              /* Normal (non-inverted) */
    send_cmd(0xD5); send_cmd(0x80); /* Clock divide */
    send_cmd(0x20); send_cmd(0x02); /* Page addressing mode */
    send_cmd(0x8D); send_cmd(0x14); /* Charge pump ON */
    send_cmd(0xAF);              /* Display ON */
}

/* ============================================================
 * ssd1306_clear
 * ============================================================ */
void ssd1306_clear(void)
{
    memset(g_fb, 0x00, sizeof(g_fb));
}

/* ============================================================
 * ssd1306_flush
 * 参考コードの oled_flush_offset() と同一
 * ============================================================ */
void ssd1306_flush(void)
{
    for (int p = 0; p < CFG_PAGES; p++) {
        send_cmd((uint8_t)(0xB0 | p));  /* page address */
        send_cmd(0x00);                  /* lower column = 0 */
        send_cmd(0x10);                  /* upper column = 0 */
        send_data(&g_fb[p * CFG_PHYS_W], CFG_PHYS_W);
    }
}

/* ============================================================
 * ssd1306_blit_gram
 *
 * BadCodec gram (1bit/px, row-major, LSB first) を
 * SSD1306 の page-mode framebuffer に変換して書き込む。
 *
 * gram のビット取得:
 *   bit_idx = y * w + x
 *   pixel   = (gram[bit_idx >> 3] >> (bit_idx & 7)) & 1
 *
 * fb への書き込み:
 *   phys_x = x + CFG_X_OFFSET
 *   phys_y = y + CFG_Y_OFFSET
 *   page   = phys_y >> 3
 *   bit    = phys_y &  7
 *   g_fb[page * CFG_PHYS_W + phys_x] |= (1 << bit)  // pixel=1
 *   g_fb[page * CFG_PHYS_W + phys_x] &= ~(1 << bit) // pixel=0
 *
 * 参考コードの draw_point() を全ピクセルに適用した等価実装。
 * ============================================================ */
void ssd1306_blit_gram(const uint8_t *gram, int w, int h)
{
    for (int y = 0; y < h; y++) {
        int phys_y = y + CFG_Y_OFFSET;
        if (phys_y < 0 || phys_y >= CFG_PHYS_H) continue;

        int page = phys_y >> 3;   /* / 8 */
        int bit  = phys_y &  7;   /* % 8 */

        for (int x = 0; x < w; x++) {
            int phys_x = x + CFG_X_OFFSET;
            if (phys_x < 0 || phys_x >= CFG_PHYS_W) continue;

            /* gram から 1bit 取得 */
            int     bi  = y * w + x;
            uint8_t pix = (gram[bi >> 3] >> (bi & 7)) & 1U;

            /* framebuffer に書き込み */
            uint8_t *cell = &g_fb[page * CFG_PHYS_W + phys_x];
            if (pix)
                *cell |=  (uint8_t)(1U << bit);
            else
                *cell &= ~(uint8_t)(1U << bit);
        }
    }
}
