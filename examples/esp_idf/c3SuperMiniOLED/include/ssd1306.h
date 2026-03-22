/**
 * @file  ssd1306.h
 * @brief SSD1306 128x64 OLED ドライバ (I2C / ESP-IDF)
 * @version v0.6.0
 *
 * 参考コードの実装を BadCodec プロジェクト用に整理したもの。
 * framebuffer (fb[128*8]) への描画と flush を提供する。
 */

#ifndef SSD1306_H
#define SSD1306_H

#include <stdint.h>
#include <stddef.h>
#include "bad_player_config.h"

/* ============================================================
 * framebuffer
 * fb[page * 128 + x]  page=0..7  x=0..127
 * bit(y%8) = pixel(x, page*8 + y%8)
 * ============================================================ */
extern uint8_t ssd1306_fb[BAD_OLED_WIDTH * BAD_OLED_PAGES];

/* ============================================================
 * API
 * ============================================================ */

/** I2C 初期化 + SSD1306 初期化シーケンス */
void ssd1306_init(void);

/** framebuffer をクリア (全0) */
void ssd1306_clear(void);

/** framebuffer を OLED に転送 */
void ssd1306_flush(void);

/**
 * BadCodec gram (1bit row-major LSB) を
 * framebuffer の (x_off, y_off) 位置に貼り付ける
 *
 * @param gram    bad_decode の gram バッファ
 * @param w       gram の幅 (= BAD_VIDEO_WIDTH)
 * @param h       gram の高さ (= BAD_VIDEO_HEIGHT)
 * @param x_off   物理画面上の X オフセット (= BAD_X_OFFSET)
 * @param y_off   物理画面上の Y オフセット (= BAD_Y_OFFSET)
 */
void ssd1306_draw_gram(const uint8_t *gram,
                       uint16_t w, uint16_t h,
                       int x_off, int y_off);

#endif /* SSD1306_H */
