/**
 * @file  ssd1306_drv.h
 * @brief SSD1306 driver for ESP32-C3 Super Mini builtin OLED
 * @version v0.6.0
 */

 #ifndef SSD1306_DRV_H
 #define SSD1306_DRV_H
 
 #include <stdint.h>
 #include "config.h"
 
 /* Framebuffer: fb[page * CFG_PHYS_W + x]
  * page = 0..7, x = 0..127
  * bit(y%8) = pixel at physical (x, page*8 + y%8)
  */
 extern uint8_t g_fb[CFG_PHYS_W * CFG_PAGES];
 
 void ssd1306_init(void);
 void ssd1306_clear(void);
 void ssd1306_flush(void);
 
 /* Copy BadCodec gram (1bit/px, LSB-first, row-major, w x h)
  * into framebuffer at (CFG_X_OFFSET, CFG_Y_OFFSET). */
 void ssd1306_blit_gram(const uint8_t *gram, int w, int h);
 
 #endif /* SSD1306_DRV_H */
 
