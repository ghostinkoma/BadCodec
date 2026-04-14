/**
 * @file  draw.c
 * @brief BadCodec 描画モジュール実装
 */

 #include "draw.h"
 #include "font.h"
 #include "ssd1306_drv.h"
 #include "config.h"
 #include "adpcm_drv.h"
 
 #include "freertos/FreeRTOS.h"
 #include "freertos/task.h"
 #include <string.h>
 #include <inttypes.h>
 #include <stdlib.h>
 #include <stdio.h>
 #include <inttypes.h>
 /* ============================================================
  * 内部: フレームバッファへの 1 ピクセル書き込み
  * ============================================================ */
 static void _pset_gram(int x, int y, color_t color)
 {
     if (x < 0 || x >= CFG_PHYS_W || y < 0 || y >= CFG_PHYS_H) return;
     int page = y >> 3;
     int bit  = y &  7;
     uint8_t *cell = &g_fb[page * CFG_PHYS_W + x];
     if (color == INVERT) {
         *cell ^= (uint8_t)(1U << bit);
     } else if (color == WHITE) {
         *cell |= (uint8_t)(1U << bit);
     } else {
         *cell &= ~(uint8_t)(1U << bit);
     }
 }
 
 /* ============================================================
  * str_t 構造体
  * ============================================================ */
 #define STR_MAX_LEN  32   /* 最大文字列長 */
 #define FRAME_MAX_STR 8   /* フレームあたり最大 str 数 */
 
 struct str_s {
     char     text[STR_MAX_LEN];
     color_t  color;
     int      x, y;
     char     scroll_dir;
     uint32_t scroll_interval_ms;
     int      scroll_pixels;
     uint32_t scroll_last_ms;
     int      active;
 };
 
 /* ============================================================
  * frame_t 構造体
  * ============================================================ */
 struct frame_s {
     str_t strs[FRAME_MAX_STR];
 };
 
 /* ============================================================
  * frame_create / frame_destroy
  * ============================================================ */
 frame_t *frame_create(void)
 {
     frame_t *f = (frame_t *)calloc(1, sizeof(frame_t));
     return f;
 }
 
 void frame_destroy(frame_t *f)
 {
     if (f) free(f);
 }
 
 /* ============================================================
  * frame_clear
  * ============================================================ */
 void frame_clear(frame_t *f)
 {
     (void)f;
     ssd1306_clear();
 }
 
 /* ============================================================
  * 基本描画プリミティブ
  * ============================================================ */
 
 void frame_pset(frame_t *f, int x, int y, color_t color)
 {
     (void)f;
     _pset_gram(x, y, color);
 }
 
 void frame_line(frame_t *f, int x0, int y0, int x1, int y1, color_t color)
 {
     (void)f;
     int dx = abs(x1 - x0), sx = (x0 < x1) ? 1 : -1;
     int dy = -abs(y1 - y0), sy = (y0 < y1) ? 1 : -1;
     int err = dx + dy;
     for (;;) {
         _pset_gram(x0, y0, color);
         if (x0 == x1 && y0 == y1) break;
         int e2 = 2 * err;
         if (e2 >= dy) { err += dy; x0 += sx; }
         if (e2 <= dx) { err += dx; y0 += sy; }
     }
 }
 
 void frame_rect(frame_t *f, int x0, int y0, int x1, int y1,
                 color_t color, int linewidth)
 {
     (void)f;
     if (x0 > x1) { int t = x0; x0 = x1; x1 = t; }
     if (y0 > y1) { int t = y0; y0 = y1; y1 = t; }
     int w = x1 - x0;
     int h = y1 - y0;
     int half = (w < h ? w : h) / 2;
     if (linewidth >= half || linewidth < 0) linewidth = half; /* 塗りつぶし */
 
     if (linewidth == 0) linewidth = 1;
 
     for (int lw = 0; lw < linewidth; lw++) {
         /* 上下 */
         for (int x = x0 + lw; x <= x1 - lw; x++) {
             _pset_gram(x, y0 + lw, color);
             _pset_gram(x, y1 - lw, color);
         }
         /* 左右 */
         for (int y = y0 + lw; y <= y1 - lw; y++) {
             _pset_gram(x0 + lw, y, color);
             _pset_gram(x1 - lw, y, color);
         }
         /* 内側を塗りつぶす場合は残り全部を埋める */
         if (lw == linewidth - 1 && linewidth == half) {
             for (int xx = x0 + lw; xx <= x1 - lw; xx++)
                 for (int yy = y0 + lw; yy <= y1 - lw; yy++)
                     _pset_gram(xx, yy, color);
         }
     }
 
     /* 完全塗りつぶし判定: linewidth >= half の場合 */
     if (linewidth >= half) {
         for (int xx = x0; xx <= x1; xx++)
             for (int yy = y0; yy <= y1; yy++)
                 _pset_gram(xx, yy, color);
     }
 }
 
 void frame_circle(frame_t *f, int cx, int cy, int r,
                   color_t color, int linewidth)
 {
     (void)f;
     if (r <= 0) { _pset_gram(cx, cy, color); return; }
     /* 塗りつぶし: linewidth >= r */
     if (linewidth < 0) linewidth = 1;
     int r_inner = (linewidth >= r) ? 0 : (r - linewidth);
 
     for (int yr = -r; yr <= r; yr++) {
         for (int xr = -r; xr <= r; xr++) {
             int dist2 = xr * xr + yr * yr;
             if (dist2 <= r * r && dist2 >= r_inner * r_inner) {
                 _pset_gram(cx + xr, cy + yr, color);
             }
         }
     }
 }
 
 /* ============================================================
  * 文字 1 文字を gram に描画
  *
  * color == INVERT の場合:
  *   前景ピクセルのみ INVERT で反転。背景は触らない。
  *   → 映像の上に透過オーバーレイとして機能する。
  *
  * color == WHITE / BLACK の場合:
  *   前景を指定色、背景を逆色（通常は BLACK）で塗る。
  * ============================================================ */
 static void _draw_char(int x, int y, char c, color_t color)
 {
     if (c < FONT_FIRST || c > FONT_LAST) c = '?';
     const uint8_t *bmp = g_font8x8[(uint8_t)c - FONT_FIRST];
     for (int row = 0; row < FONT_ROWS; row++) {
         uint8_t line = bmp[row];
         for (int col = 0; col < FONT_W; col++) {
             int px = x + col, py = y + row;
             if (px < 0 || px >= CFG_PHYS_W || py < 0 || py >= CFG_PHYS_H) continue;
             if (line & (0x80U >> col)) {
                 /* 前景ピクセル: 指定色で描画 */
                 _pset_gram(px, py, color);
             } else {
                 /* 背景ピクセル:
                  * INVERT モードでは背景を触らない (透過オーバーレイ)。
                  * WHITE/BLACK モードでは背景を BLACK で塗る。          */
                 if (color != INVERT) {
                     _pset_gram(px, py, BLACK);
                 }
             }
         }
     }
 }
 
 /* ============================================================
  * str_t API
  * ============================================================ */
 str_t *frame_print(frame_t *f, const char *text, color_t color)
 {
     for (int i = 0; i < FRAME_MAX_STR; i++) {
         str_t *s = &f->strs[i];
         if (!s->active) {
             s->active   = 1;
             s->color    = color;
             s->x = s->y = 0;
             s->scroll_dir = 0;
             s->scroll_interval_ms = 0;
             s->scroll_pixels = 0;
             s->scroll_last_ms = 0;
             strncpy(s->text, text, STR_MAX_LEN - 1);
             s->text[STR_MAX_LEN - 1] = '\0';
             return s;
         }
     }
     return NULL; /* slots full */
 }
 
 void str_position(str_t *s, int x, int y)
 {
     if (!s) return;
     s->x = x;
     s->y = y;
 }
 
 void str_move(str_t *s, int dx, int dy)
 {
     if (!s) return;
     s->x += dx;
     s->y += dy;
 }
 
 void str_scroll(str_t *s, char dir, uint32_t interval_ms, int pixels)
 {
     if (!s) return;
     s->scroll_dir         = dir;
     s->scroll_interval_ms = interval_ms;
     s->scroll_pixels      = pixels;
 }
 
 /* ============================================================
  * str_t を gram に描画 (frame_flash 内部から呼ぶ)
  * ============================================================ */
 static void _str_render(str_t *s)
 {
     if (!s || !s->active) return;
 
     /* スクロール処理 */
     if (s->scroll_dir && s->scroll_interval_ms > 0) {
         uint32_t now = (uint32_t)(xTaskGetTickCount() * portTICK_PERIOD_MS);
         if (now - s->scroll_last_ms >= s->scroll_interval_ms) {
             s->scroll_last_ms = now;
             if (s->scroll_dir == SCROLL_H) s->x -= s->scroll_pixels;
             else                           s->y -= s->scroll_pixels;
         }
     }
 
     /* 描画 */
     int cx = s->x;
     for (const char *p = s->text; *p; p++, cx += FONT_W) {
         _draw_char(cx, s->y, *p, s->color);
     }
 }
 
 /* ============================================================
  * frame_flash
  * ============================================================ */
 void frame_flash(frame_t *f)
 {
     if (!f) return;
     /* str オブジェクトを gram に描画 */
     for (int i = 0; i < FRAME_MAX_STR; i++) {
         _str_render(&f->strs[i]);
     }
     /* OLED に転送 */
     ssd1306_flush();
 }
 
 /* ============================================================
  * OSD 描画
  * ============================================================
  *
  * 全OSD要素は INVERT (NOT) オーバーレイで描画する。
  *
  * 設計原則:
  *   - ssd1306_clear() + ssd1306_blit_gram() で映像フレームが毎回
  *     フレームバッファに書き込まれる。
  *   - その後 frame_osd_update() が INVERT で上書きする。
  *   - INVERT はピクセルの現在値を反転するだけなので、
  *     映像ピクセルが白なら黒テキスト、黒なら白テキストになり
  *     常に視認可能なオーバーレイになる。
  *   - 前フレームの OSD 残像は映像フレームの ssd1306_clear() で
  *     自動的に消えるため、差分更新や消去処理が不要。
  *
  * VU バー:
  *   毎フレーム (frame_osd_update() 呼び出しごとに) バーを
  *   INVERT で全幅描画する。状態変数不要、チラつきなし。
  *   "VU" ラベルも同様。
  * ============================================================ */

 /* OSD 行 Y 座標 (config.h で変更可能) */
 #define OSD_ROW_CPU   CFG_OSD_Y_CPU
 #define OSD_ROW_FPS   CFG_OSD_Y_FPS
 #define OSD_ROW_VU    CFG_OSD_Y_VU

 /* VU バーグラフのサイズ */
 #define VU_BAR_X     24   /* "VU " ラベル (3文字×8px=24) の直後 */
 #define VU_BAR_W    104   /* 128 - 24 = 104px */
 #define VU_BAR_H      6   /* バー高さ (px) */

 void frame_osd_update(frame_t *f,
                       uint32_t fps_x10,
                       uint32_t cpu_x10,
                       uint16_t vu_raw)
 {
     if (!f) return;

     /* --- VU (INVERT オーバーレイ、毎フレーム描画) --- */
 #if (CFG_OSD_VU == 1)
     {
         /* "VU" ラベル */
         _draw_char(0,      OSD_ROW_VU, 'V', INVERT);
         _draw_char(FONT_W, OSD_ROW_VU, 'U', INVERT);

         /* バー幅を計算: vu_raw [0,32767] → [0, VU_BAR_W] */
         int fill = (int)((uint32_t)vu_raw * (uint32_t)VU_BAR_W / 32767U);
         if (fill > VU_BAR_W) { fill = VU_BAR_W; }

         /* 有効部分を INVERT で描画。
          * 映像フレームが ssd1306_clear() で黒くリセットされているため
          * INVERT = WHITE になり、バーが白く光って見える。
          * 映像があるピクセルは反転されるので常に視認可能。         */
         for (int x = VU_BAR_X; x < VU_BAR_X + fill; x++) {
             for (int y = OSD_ROW_VU; y < OSD_ROW_VU + VU_BAR_H; y++) {
                 _pset_gram(x, y, INVERT);
             }
         }
     }
 #else
     (void)vu_raw;
 #endif /* CFG_OSD_VU */

     /* --- FPS / CPU (INVERT オーバーレイ) --- */
 #if (CFG_OSD_FPS == 1) || (CFG_OSD_CPU == 1)
     char buf[24];
 #endif

 #if (CFG_OSD_FPS == 1)
     snprintf(buf, sizeof(buf), "FPS%3u.%u",
              (unsigned)(fps_x10 / 10U), (unsigned)(fps_x10 % 10U));
     for (int i = 0; buf[i]; i++) {
         _draw_char(i * FONT_W, OSD_ROW_FPS, buf[i], INVERT);
     }
 #else
     (void)fps_x10;
 #endif

 #if (CFG_OSD_CPU == 1)
     snprintf(buf, sizeof(buf), "CPU%3u.%u%%",
              (unsigned)(cpu_x10 / 10U), (unsigned)(cpu_x10 % 10U));
     for (int i = 0; buf[i]; i++) {
         _draw_char(i * FONT_W, OSD_ROW_CPU, buf[i], INVERT);
     }
 #else
     (void)cpu_x10;
 #endif
 }
 
