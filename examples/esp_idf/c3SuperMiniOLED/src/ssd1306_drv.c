/**
 * @file  ssd1306_drv.c
 * @brief SSD1306 driver implementation
 * @version v0.6.0
 *
 * Ported directly from reference code (old main.c).
 */

 #include "ssd1306_drv.h"
 #include "driver/i2c.h"
 #include "freertos/FreeRTOS.h"
 #include "freertos/task.h"
 #include <string.h>
 
 uint8_t g_fb[CFG_PHYS_W * CFG_PAGES];
 
 /* ---- low level ---- */
 
 static void send_cmd(uint8_t cmd)
 {
     i2c_cmd_handle_t h = i2c_cmd_link_create();
     i2c_master_start(h);
     i2c_master_write_byte(h, (CFG_OLED_ADDR << 1) | I2C_MASTER_WRITE, true);
     i2c_master_write_byte(h, 0x00, true);
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
     i2c_master_write_byte(h, 0x40, true);
     i2c_master_write(h, data, len, true);
     i2c_master_stop(h);
     i2c_master_cmd_begin(CFG_I2C_PORT, h, pdMS_TO_TICKS(100));
     i2c_cmd_link_delete(h);
 }
 
 /* ---- public API ---- */
 
 void ssd1306_init(void)
 {
     /* I2C master init */
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
 
     /* SSD1306 init sequence (same as reference code) */
     send_cmd(0xAE);
     send_cmd(0xA8); send_cmd(0x3F);
     send_cmd(0xD3); send_cmd(0x00);
     send_cmd(0x40);
     send_cmd(0xA1);
     send_cmd(0xC8);
     send_cmd(0xDA); send_cmd(0x12);
     send_cmd(0x81); send_cmd(0x7F);
     send_cmd(0xA4);
     send_cmd(0xA6);
     send_cmd(0xD5); send_cmd(0x80);
     send_cmd(0x20); send_cmd(0x02);
     send_cmd(0x8D); send_cmd(0x14);
     send_cmd(0xAF);
 }
 
 void ssd1306_clear(void)
 {
     memset(g_fb, 0x00, sizeof(g_fb));
 }
 
 /* Same as oled_flush_offset() in reference code */
 void ssd1306_flush(void)
 {
     for (int p = 0; p < CFG_PAGES; p++) {
         send_cmd((uint8_t)(0xB0 | p));
         send_cmd(0x00);
         send_cmd(0x10);
         send_data(&g_fb[p * CFG_PHYS_W], (size_t)CFG_PHYS_W);
     }
 }
 
 /*
  * ssd1306_blit_gram
  *
  * Converts BadCodec gram (1bit/px, LSB-first, row-major)
  * to SSD1306 page-mode framebuffer and writes at offset.
  *
  * gram bit access:
  *   bit_idx = y*w + x
  *   pixel   = (gram[bit_idx>>3] >> (bit_idx&7)) & 1
  *
  * fb write:
  *   phys_x = x + CFG_X_OFFSET
  *   phys_y = y + CFG_Y_OFFSET
  *   page   = phys_y >> 3
  *   bit    = phys_y &  7
  *
  * Equivalent to calling draw_point(x,y) for every pixel.
  */
 void ssd1306_blit_gram(const uint8_t *gram, int w, int h)
 {
     int x, y;
     for (y = 0; y < h; y++) {
         int phys_y = y + CFG_Y_OFFSET;
         if (phys_y < 0 || phys_y >= CFG_PHYS_H) continue;
         int page = phys_y >> 3;
         int bit  = phys_y &  7;
 
         for (x = 0; x < w; x++) {
             int phys_x = x + CFG_X_OFFSET;
             if (phys_x < 0 || phys_x >= CFG_PHYS_W) continue;
 
             int     bi  = y * w + x;
             uint8_t pix = (gram[bi >> 3] >> (bi & 7)) & 1U;
             uint8_t *cell = &g_fb[page * CFG_PHYS_W + phys_x];
             if (pix)
                 *cell |=  (uint8_t)(1U << bit);
             else
                 *cell &= ~(uint8_t)(1U << bit);
         }
     }
 }
 
