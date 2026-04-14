/**
 * @file  draw.h
 * @brief BadCodec 描画モジュール
 *
 * SSD1306 フレームバッファへの描画 API。
 *
 * 使い方:
 *   frame_t *f = frame_create();    // フレームオブジェクト生成
 *   frame_pset(f, 10, 20, WHITE);   // 点
 *   frame_line(f, 0,0, 127,63, WHITE);
 *   frame_rect(f, 10,10, 50,30, WHITE, 1);
 *   frame_circle(f, 64,32, 20, WHITE, 1);
 *   str_t *s = frame_print(f, "Hello", WHITE);
 *   str_position(s, 0, 0);
 *   frame_flash(f);                 // OLED に転送
 *   frame_destroy(f);
 *
 * color_t:
 *   WHITE  = 1 (白: ピクセルを点灯)
 *   BLACK  = 0 (黒: ピクセルを消灯)
 *   INVERT = 2 (現在の gram を NOT して書き込む)
 */

 #ifndef DRAW_H
 #define DRAW_H
 
 #include <stdint.h>
 #include "config.h"
 
 /* ---- 色定数 ---------------------------------------------- */
 typedef uint8_t color_t;
 #define BLACK   ((color_t)0)
 #define WHITE   ((color_t)1)
 #define INVERT  ((color_t)2)
 
 /* ---- フォワード宣言 --------------------------------------- */
 typedef struct frame_s frame_t;
 typedef struct str_s   str_t;
 
 /* ============================================================
  * frame_t — フレームオブジェクト
  *
  * フレームは描画コマンドのキューを持つ。
  * frame_flash() を呼ぶと一括して SSD1306 に転送する。
  * ============================================================ */
 
 /** フレームオブジェクト生成。失敗時は NULL。 */
 frame_t *frame_create(void);
 
 /** フレームオブジェクト破棄。 */
 void frame_destroy(frame_t *f);
 
 /** フレームに登録されたすべてのオブジェクトを gram に描画し OLED へ転送。 */
 void frame_flash(frame_t *f);
 
 /** gram を黒でクリア (frame_flash の前処理として使う)。 */
 void frame_clear(frame_t *f);
 
 /* ---- 基本描画 (即時: frame_flash で反映) --------------- */
 
 /** 点を描画。 */
 void frame_pset(frame_t *f, int x, int y, color_t color);
 
 /** 直線 (Bresenham)。 */
 void frame_line(frame_t *f, int x0, int y0, int x1, int y1, color_t color);
 
 /**
  * 矩形。
  * linewidth=1 : 外周のみ
  * linewidth>=r (rectangle の場合は min(w,h)/2) : 塗りつぶし
  * それ以外: linewidth ドット分の枠線
  */
 void frame_rect(frame_t *f,
                 int x0, int y0, int x1, int y1,
                 color_t color, int linewidth);
 
 /**
  * 円。
  * @param cx,cy  中心座標
  * @param r      半径
  * @param color  色
  * @param linewidth  線幅 (r を指定すると塗りつぶし)
  */
 void frame_circle(frame_t *f,
                   int cx, int cy, int r,
                   color_t color, int linewidth);
 
 /* ============================================================
  * str_t — 文字列オブジェクト
  *
  * frame_print() で生成し、frame_flash() 時に gram に書かれる。
  * ============================================================ */
 
 /**
  * 文字列オブジェクトを生成してフレームに登録する。
  * @param f      フレーム
  * @param text   ASCII 文字列 (NUL 終端)
  * @param color  描画色
  * @return str_t* (失敗時 NULL)
  */
 str_t *frame_print(frame_t *f, const char *text, color_t color);
 
 /** 文字列の表示位置を設定。 */
 void str_position(str_t *s, int x, int y);
 
 /** 文字列を移動 (現在位置からの相対移動)。 */
 void str_move(str_t *s, int dx, int dy);
 
 /**
  * スクロール設定。
  * @param s         str_t
  * @param dir       'h' または 'v'  (SCROLL_H / SCROLL_V)
  * @param interval  スクロール間隔 ms
  * @param pixels    1回あたりの移動量 (ドット)
  *
  * frame_flash() を呼ぶたびに interval ms 経過していれば
  * pixels ドット移動する。
  */
 #define SCROLL_H  'h'
 #define SCROLL_V  'v'
 void str_scroll(str_t *s, char dir, uint32_t interval_ms, int pixels);
 
 /* ============================================================
  * OSD (On-Screen Display) — CPU/FPS/VU
  *
  * 最下段から: CPU → FPS → VU の順で表示。
  * frame_flash() の直前に frame_osd_update() を呼ぶ。
  * ============================================================ */
 
 /**
  * OSD を更新して gram に書き込む。
  * @param f      フレーム
  * @param fps    実測 FPS (× 10, 例: 34.5fps → 345)
  * @param cpu    CPU 使用率 (× 10, 例: 43.2% → 432)
  * @param vu_raw adpcm_get_vu() の戻り値 (0〜32767)
  */
 void frame_osd_update(frame_t *f,
                       uint32_t fps_x10,
                       uint32_t cpu_x10,
                       uint16_t vu_raw);
 
 #endif /* DRAW_H */
 
