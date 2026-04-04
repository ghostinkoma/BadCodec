/**
 * @file  ssd1306_drv.c
 * @brief SSD1306 128x64 — ESP32-C3 Super Mini, external module, 3.3V
 * @version v0.6.3
 *
 * Root cause of "blank display" fixed in this version:
 *
 *   Previous version sent 0x20 0x00 (Horizontal Addressing Mode) in init,
 *   but ssd1306_flush() used 0xB0 / 0x00 / 0x10 which are Page Addressing
 *   Mode ONLY commands.  The SSD1306 datasheet states explicitly:
 *     "Set Lower/Higher Column Start Address for Page Addressing Mode:
 *      This command is only for page addressing mode."
 *   In Horizontal mode those commands are silently ignored, so the GRAM
 *   write pointer never resets between pages → garbled / blank display.
 *
 *   Fix: change 0x20 0x00 → 0x20 0x02 (Page Addressing Mode).
 *   ssd1306_flush() already uses the correct Page-mode sequence.
 *
 * Other settings (all correct, kept unchanged):
 *   0x8D 0x14  Charge pump ENABLE  (required for 3.3V single supply)
 *   0xD9 0xF1  Pre-charge period   (required for charge pump / 3.3V)
 *   0xDB 0x40  VCOMH deselect      (required for charge pump / 3.3V)
 *   0x2E       Deactivate scroll   (some modules boot with scroll active)
 *   110ms delay after 0xAF        (datasheet tAF = 100ms min)
 *
 * GPIO8 / GPIO9 note:
 *   These are boot-strapping pins but are safe for I2C when NO external
 *   pull-up resistors are fitted.  The internal pull-ups (~45 kΩ) enabled
 *   below are too weak to affect the boot strapping level.
 */

 #include "ssd1306_drv.h"
 #include "driver/i2c.h"
 #include "freertos/FreeRTOS.h"
 #include "freertos/task.h"
 #include "esp_log.h"
 #include <string.h>

 static const char *TAG = "SSD1306";
 
 uint8_t g_fb[CFG_PHYS_W * CFG_PAGES];
 
 /* ============================================================
  * Low-level I2C helpers
  * ============================================================ */
 
 static void send_cmd(uint8_t cmd)
 {
     i2c_cmd_handle_t h = i2c_cmd_link_create();
     i2c_master_start(h);
     i2c_master_write_byte(h, (CFG_OLED_ADDR << 1) | I2C_MASTER_WRITE, true);
     i2c_master_write_byte(h, 0x00, true);   /* Co=0, D/C#=0 → command */
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
     i2c_master_write_byte(h, 0x40, true);   /* Co=0, D/C#=1 → data */
     i2c_master_write(h, data, len, true);
     i2c_master_stop(h);
     i2c_master_cmd_begin(CFG_I2C_PORT, h, pdMS_TO_TICKS(100));
     i2c_cmd_link_delete(h);
 }
 
 /* ============================================================
  * ssd1306_init
  *
  * Sequence for SSD1306 128x64, 3.3V single supply,
  * internal charge pump.
  * ============================================================ */
 void ssd1306_init(void)
 {
     /* I2C master */
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
 
     ESP_LOGI(TAG, "SDA=GPIO%d SCL=GPIO%d %dkHz addr=0x%02X",
              CFG_I2C_SDA, CFG_I2C_SCL,
              CFG_I2C_FREQ_HZ / 1000, CFG_OLED_ADDR);
 
     send_cmd(0xAE);                 /* 1.  Display OFF                    */
 
     send_cmd(0xD5); send_cmd(0x80); /* 2.  Clock: div=1, freq=8           */
 
     send_cmd(0xA8); send_cmd(0x3F); /* 3.  MUX ratio = 64                 */
 
     send_cmd(0xD3); send_cmd(0x00); /* 4.  Display offset = 0             */
 
     send_cmd(0x40);                 /* 5.  Start line = 0                 */
 
     send_cmd(0x8D); send_cmd(0x14); /* 6.  Charge pump ON                 */
                                     /*     REQUIRED for 3.3V operation    */
 
     send_cmd(0x20); send_cmd(0x02); /* 7.  *** PAGE ADDRESSING MODE ***   */
                                     /*     0x02 = Page mode               */
                                     /*     MUST match ssd1306_flush()     */
                                     /*     which uses 0xB0/0x00/0x10      */
                                     /*     (those cmds are page-mode only) */
 
     send_cmd(0xA1);                 /* 8.  SEG remap: col127→SEG0         */
                                     /*     swap to 0xA0 if L/R mirrored   */
 
     send_cmd(0xC8);                 /* 9.  COM scan: reverse              */
                                     /*     swap to 0xC0 if upside-down    */
 
     send_cmd(0xDA); send_cmd(0x12); /* 10. COM pins: alt, no remap        */
                                     /*     0x12 for 128x64                */
 
     send_cmd(0x81); send_cmd(0xCF); /* 11. Contrast = 0xCF (3.3V)        */
 
     send_cmd(0xD9); send_cmd(0xF1); /* 12. Pre-charge = 0xF1 (3.3V)      */
                                     /*     REQUIRED for 3.3V / chg pump  */
 
     send_cmd(0xDB); send_cmd(0x40); /* 13. VCOMH deselect = 0x40         */
                                     /*     REQUIRED for 3.3V / chg pump  */
 
     send_cmd(0xA4);                 /* 14. Output follows RAM             */
 
     send_cmd(0xA6);                 /* 15. Normal display (not inverted)  */
 
     send_cmd(0x2E);                 /* 16. Deactivate scroll              */
 
     send_cmd(0xAF);                 /* 17. Display ON                     */
 
     /* Datasheet tAF: SEG/COM become active 100ms after Display ON.
      * Must not call ssd1306_flush() before this window expires.   */
     vTaskDelay(pdMS_TO_TICKS(110));
 
     ESP_LOGI(TAG, "init done (128x64, page mode, charge pump ON)");
 }
 
 /* ============================================================
  * ssd1306_clear
  * ============================================================ */
 void ssd1306_clear(void)
 {
     memset(g_fb, 0x00, sizeof(g_fb));
 }
 
 /* ============================================================
  * ssd1306_flush  — Page Addressing Mode
  *
  * For each of 8 pages:
  *   0xB0|p  Set page address          (Page mode only)
  *   0x00    Lower column start = 0    (Page mode only)
  *   0x10    Upper column start = 0    (Page mode only)
  *   data    128 bytes
  *
  * This is the canonical Page-mode flush sequence.
  * It matches the Memory Mode set in ssd1306_init() (0x20 0x02).
  * ============================================================ */
 void ssd1306_flush(void)
 {
     for (int p = 0; p < CFG_PAGES; p++) {
         send_cmd((uint8_t)(0xB0 | p));
         send_cmd(0x00);
         send_cmd(0x10);
         send_data(&g_fb[p * CFG_PHYS_W], (size_t)CFG_PHYS_W);
     }
 }
 
 /* ============================================================
  * ssd1306_blit_gram
  *
  * BadCodec gram (1-bit/px, LSB-first, row-major, w×h)
  * → SSD1306 page-mode framebuffer.
  *
  * CFG_X_OFFSET=0, CFG_Y_OFFSET=0 (128×64 full panel):
  *   gram(x, y) → fb[page=y>>3][col=x]  bit = y & 7
  * ============================================================ */
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
 
