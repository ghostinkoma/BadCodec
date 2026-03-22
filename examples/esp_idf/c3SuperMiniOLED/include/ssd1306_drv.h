/**
 * @file  ssd1306_drv.h
 * @brief SSD1306 ドライバ (ESP32-C3 Super Mini / I2C)
 * @version v0.6.0
 */

#ifndef SSD1306_DRV_H
#define SSD1306_DRV_H

#include <stdint.h>
#include "config.h"

/** framebuffer: fb[page * CFG_PHYS_W + x] */
extern uint8_t g_fb[CFG_PHYS_W * CFG_PAGES];

/** I2C 初期化 + SSD1306 初期化シーケンス送信 */
void ssd1306_init(void);

/** framebuffer を 0 クリア */
void ssd1306_clear(void);

/** framebuffer を OLED に転送 */
void ssd1306_flush(void);

/**
 * BadCodec gram を framebuffer に書き込む
 *
 * gram 形式: 1bit/pixel  LSB first  row-major
 *   pixel(x,y) = (gram[(y*w + x) >> 3] >> ((y*w + x) & 7)) & 1
 *
 * @param gram  bad_decode の gram バッファ
 * @param w     gram 幅 (= CFG_VIDEO_W = 72)
 * @param h     gram 高さ (= CFG_VIDEO_H = 40)
 */
void ssd1306_blit_gram(const uint8_t *gram, int w, int h);

#endif /* SSD1306_DRV_H */
