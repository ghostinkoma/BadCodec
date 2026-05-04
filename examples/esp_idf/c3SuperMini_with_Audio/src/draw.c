/**
 * @file  draw.c
 * @brief BadCodec 描画モジュール v3.0.0
 *
 * OSD レイヤー設計:
 *   s_osd_layer[1024] を独立したフレームバッファとして保持。
 *   映像フレームを g_fb に書いた後、s_osd_layer の各ビットを
 *   NOT (INVERT) で g_fb に重畳する。
 *   これにより映像への OSD 書き込みを完全に分離でき、
 *   ボタン押下時の画面乱れが発生しない。
 *
 * OSD 描画フロー:
 *   1. osd_layer_clear()        OSDレイヤーをゼロクリア
 *   2. osd_draw_*(...)          OSDレイヤーへ描画 (WHITE のみ使用)
 *   3. osd_layer_blit()         OSDレイヤーを g_fb に INVERT で重畳
 *   4. ssd1306_flush()          OLED へ転送
 *
 * VU: ISR で更新された s_vu_peak 値をそのまま使用 (旧方式に戻す)
 */

 #include "draw.h"
 #include "font.h"
 #include "ssd1306_drv.h"
 #include "config.h"
 #include "adpcm_drv.h"

 #include "freertos/FreeRTOS.h"
 #include "freertos/task.h"
 #include <string.h>
 #include <stdlib.h>
 #include <stdio.h>
 #include <inttypes.h>

/* ============================================================
 * OSD レイヤーバッファ (映像とは独立した 1bit/px バッファ)
 * SSD1306 と同じページ形式: [page * 128 + x] bit(y%8)
 * ============================================================ */
#define OSD_BUF_SIZE  (CFG_PHYS_W * CFG_PAGES)  /* 128*8 = 1024 bytes */
static uint8_t s_osd_layer[OSD_BUF_SIZE];

/** OSD レイヤーをクリア */
void osd_layer_clear(void)
{
    memset(s_osd_layer, 0x00, sizeof(s_osd_layer));
}

/** OSD レイヤーに 1 ピクセル書き込む (WHITE のみ有効) */
static void _osd_pset(int x, int y)
{
    if (x < 0 || x >= CFG_PHYS_W || y < 0 || y >= CFG_PHYS_H) { return; }
    int page = y >> 3;
    int bit  = y &  7;
    s_osd_layer[page * CFG_PHYS_W + x] |= (uint8_t)(1U << bit);
}

/** OSD レイヤーを g_fb に INVERT で重畳する
 *  OSD が 1 のピクセルだけ g_fb を反転 → 映像に透過で重なる */
void osd_layer_blit(void)
{
    for (int i = 0; i < OSD_BUF_SIZE; i++) {
        g_fb[i] ^= s_osd_layer[i];
    }
}

/* ============================================================
 * OSD レイヤーへの文字描画
 * ============================================================ */
static void _osd_char(int x, int y, char c)
{
    if (c < FONT_FIRST || c > FONT_LAST) { c = '?'; }
    const uint8_t *bmp = g_font8x8[(uint8_t)c - FONT_FIRST];
    for (int row = 0; row < FONT_ROWS; row++) {
        uint8_t line = bmp[row];
        for (int col = 0; col < FONT_W; col++) {
            if (line & (0x80U >> col)) {
                _osd_pset(x + col, y + row);
            }
        }
    }
}

static void _osd_str(int x, int y, const char *s)
{
    for (; *s; s++, x += FONT_W) {
        _osd_char(x, y, *s);
    }
}

/* ============================================================
 * g_fb への直接描画 (映像レイヤー)
 * ============================================================ */
static void _pset_gram(int x, int y, color_t color)
{
    if (x < 0 || x >= CFG_PHYS_W || y < 0 || y >= CFG_PHYS_H) { return; }
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
 * str_t / frame_t (既存互換)
 * ============================================================ */
#define STR_MAX_LEN   32
#define FRAME_MAX_STR  8

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
struct frame_s { str_t strs[FRAME_MAX_STR]; };

frame_t *frame_create(void)  { return (frame_t *)calloc(1, sizeof(frame_t)); }
void frame_destroy(frame_t *f) { if (f) free(f); }
void frame_clear(frame_t *f)   { (void)f; ssd1306_clear(); }

void frame_pset(frame_t *f, int x, int y, color_t color)
{ (void)f; _pset_gram(x, y, color); }

void frame_line(frame_t *f, int x0, int y0, int x1, int y1, color_t color)
{
    (void)f;
    int dx=abs(x1-x0), sx=(x0<x1)?1:-1;
    int dy=-abs(y1-y0), sy=(y0<y1)?1:-1;
    int err=dx+dy;
    for(;;){
        _pset_gram(x0,y0,color);
        if(x0==x1&&y0==y1) break;
        int e2=2*err;
        if(e2>=dy){ err+=dy; x0+=sx; }
        if(e2<=dx){ err+=dx; y0+=sy; }
    }
}

void frame_rect(frame_t *f, int x0, int y0, int x1, int y1,
                color_t color, int lw)
{
    (void)f;
    if(x0>x1){int t=x0;x0=x1;x1=t;}
    if(y0>y1){int t=y0;y0=y1;y1=t;}
    int w=x1-x0, h=y1-y0;
    int half=(w<h?w:h)/2;
    if(lw>=half||lw<0){ lw=half; }
    if(lw==0){ lw=1; }
    for(int l=0;l<lw;l++){
        for(int x=x0+l;x<=x1-l;x++){
            _pset_gram(x,y0+l,color);
            _pset_gram(x,y1-l,color);
        }
        for(int y=y0+l;y<=y1-l;y++){
            _pset_gram(x0+l,y,color);
            _pset_gram(x1-l,y,color);
        }
    }
    if(lw>=half){
        for(int xx=x0;xx<=x1;xx++)
            for(int yy=y0;yy<=y1;yy++)
                _pset_gram(xx,yy,color);
    }
}

void frame_circle(frame_t *f, int cx, int cy, int r, color_t color, int lw)
{
    (void)f;
    if(r<=0){_pset_gram(cx,cy,color);return;}
    if(lw<0){lw=1;}
    int ri=(lw>=r)?0:(r-lw);
    for(int yr=-r;yr<=r;yr++){
        for(int xr=-r;xr<=r;xr++){
            int d2=xr*xr+yr*yr;
            if(d2<=r*r && d2>=ri*ri){
                _pset_gram(cx+xr,cy+yr,color);
            }
        }
    }
}

static void _draw_char(int x, int y, char c, color_t color)
{
    if(c<FONT_FIRST||c>FONT_LAST){ c='?'; }
    const uint8_t *bmp=g_font8x8[(uint8_t)c-FONT_FIRST];
    for(int row=0;row<FONT_ROWS;row++){
        uint8_t line=bmp[row];
        for(int col=0;col<FONT_W;col++){
            int px=x+col, py=y+row;
            if(px<0||px>=CFG_PHYS_W||py<0||py>=CFG_PHYS_H) continue;
            if(line&(0x80U>>col)){
                _pset_gram(px,py,color);
            } else if(color!=INVERT){
                _pset_gram(px,py,BLACK);
            }
        }
    }
}

str_t *frame_print(frame_t *f, const char *text, color_t color)
{
    for(int i=0;i<FRAME_MAX_STR;i++){
        str_t *s=&f->strs[i];
        if(!s->active){
            s->active=1; s->color=color;
            s->x=s->y=0; s->scroll_dir=0;
            s->scroll_interval_ms=s->scroll_pixels=s->scroll_last_ms=0;
            strncpy(s->text,text,STR_MAX_LEN-1);
            s->text[STR_MAX_LEN-1]='\0';
            return s;
        }
    }
    return NULL;
}

void str_position(str_t *s, int x, int y){ if(s){s->x=x;s->y=y;} }
void str_move(str_t *s, int dx, int dy)  { if(s){s->x+=dx;s->y+=dy;} }
void str_scroll(str_t *s, char dir, uint32_t iv, int px)
{
    if(!s) return;
    s->scroll_dir=dir; s->scroll_interval_ms=iv; s->scroll_pixels=px;
}

static void _str_render(str_t *s)
{
    if(!s||!s->active) return;
    if(s->scroll_dir && s->scroll_interval_ms>0){
        uint32_t now=(uint32_t)(xTaskGetTickCount()*portTICK_PERIOD_MS);
        if(now-s->scroll_last_ms>=s->scroll_interval_ms){
            s->scroll_last_ms=now;
            if(s->scroll_dir==SCROLL_H){ s->x-=s->scroll_pixels; }
            else                        { s->y-=s->scroll_pixels; }
        }
    }
    int cx=s->x;
    for(const char *p=s->text;*p;p++,cx+=FONT_W){
        _draw_char(cx,s->y,*p,s->color);
    }
}

void frame_flash(frame_t *f)
{
    if(!f) return;
    for(int i=0;i<FRAME_MAX_STR;i++){ _str_render(&f->strs[i]); }
    ssd1306_flush();
}

/* ============================================================
 * OSD メイン描画
 *
 * 呼び出し順:
 *   ssd1306_clear();
 *   ssd1306_blit_gram(...);   ← 映像フレームを g_fb に書く
 *   osd_layer_clear();
 *   frame_osd_update(...);    ← OSD レイヤーに描画
 *   osd_layer_blit();         ← OSD を g_fb に INVERT 重畳
 *   ssd1306_flush();
 *
 * VU バー:
 *   旧方式に戻す: adpcm_get_vu() の戻り値 [0-32767] をそのまま使用。
 *   表示: 上部 y=8 に "VU |||||||||||||" 形式で 96px 幅バー。
 *   96px = 2^n に近い → fill = vu_raw * 96 / 32768 = vu_raw >> 9 * 3
 *   実際: fill = (vu_raw * 96) >> 15  (割り算なし、シフトのみ)
 *
 * FPS/CPU: 旧来通り OSD レイヤーに白文字で描画。
 * VOL 表示: vol_show フラグが立っているとき最上段に表示。
 * PAUSE 表示: paused フラグが立っているとき中央に点滅表示。
 * ============================================================ */

/* OSD Y 座標 (config.h で上書き可) */
#ifndef CFG_OSD_Y_CPU
#define CFG_OSD_Y_CPU  56
#endif
#ifndef CFG_OSD_Y_FPS
#define CFG_OSD_Y_FPS  48
#endif
#ifndef CFG_OSD_Y_VU
#define CFG_OSD_Y_VU   40
#endif

/* VU バー: 上部に配置、96px 幅 */
#define VU_LABEL_W    (2 * FONT_W)   /* "VU" 2文字 = 16px */
#define VU_BAR_X2     (VU_LABEL_W + 4)  /* バー開始 X */
#define VU_BAR_W2     96                  /* バー幅 (2^n 近似でシフト演算可) */
#define VU_BAR_H2      5                  /* バー高さ */
/* fill = vu_raw * 96 / 32768 ≒ vu_raw * 3 >> 10 */

/* PAUSE 点滅: 500ms 周期 */
#define PAUSE_BLINK_MS  500U

void frame_osd_update(frame_t *f,
                      uint32_t fps_x10,
                      uint32_t cpu_x10,
                      uint16_t vu_raw)
{
    (void)f;
    char buf[24];

    /* --- VU (OSD レイヤーへ描画) --- */
#if (CFG_OSD_VU == 1)
    {
        /* "VU" ラベル */
        _osd_char(0,        CFG_OSD_Y_VU, 'V');
        _osd_char(FONT_W,   CFG_OSD_Y_VU, 'U');

        /* バー幅: vu_raw * 96 >> 15 (割り算なし) */
        int fill = (int)(((uint32_t)vu_raw * 96U) >> 15);
        if (fill > VU_BAR_W2) { fill = VU_BAR_W2; }

        for (int x = VU_BAR_X2; x < VU_BAR_X2 + fill; x++) {
            for (int y = CFG_OSD_Y_VU; y < CFG_OSD_Y_VU + VU_BAR_H2; y++) {
                _osd_pset(x, y);
            }
        }
    }
#else
    (void)vu_raw;
#endif

    /* --- FPS (OSD レイヤーへ描画) --- */
#if (CFG_OSD_FPS == 1)
    if (fps_x10 != 0xFFFFFFFFUL) {
        snprintf(buf, sizeof(buf), "FPS%3u.%u",
                 (unsigned)(fps_x10/10U), (unsigned)(fps_x10%10U));
        _osd_str(0, CFG_OSD_Y_FPS, buf);
    }
#else
    (void)fps_x10;
#endif

    /* --- CPU (OSD レイヤーへ描画) --- */
#if (CFG_OSD_CPU == 1)
    if (cpu_x10 != 0xFFFFFFFFUL) {
        snprintf(buf, sizeof(buf), "CPU%3u.%u%%",
                 (unsigned)(cpu_x10/10U), (unsigned)(cpu_x10%10U));
        _osd_str(0, CFG_OSD_Y_CPU, buf);
    }
#else
    (void)cpu_x10;
#endif
}

/* ============================================================
 * VOL 表示 (最上段 y=0)
 * "VOL |||||||||||     " 形式で 16段階バー
 * ============================================================ */
void osd_draw_vol(uint8_t vol_step)
{
    /* "VOL " ラベル */
    _osd_char(0,             0, 'V');
    _osd_char(FONT_W,        0, 'O');
    _osd_char(FONT_W * 2,    0, 'L');
    _osd_char(FONT_W * 3,    0, ' ');

    /* バー: vol_step/15 × 80px (FONT_W*10) */
    /* バー幅 = vol_step * 80 / 15 ≒ vol_step * 5 + vol_step/3 */
    int bar_max = 80;
    int fill    = (vol_step == 0) ? 0 : (int)((uint32_t)vol_step * bar_max / 15U);
    if (fill > bar_max) { fill = bar_max; }

    int bx = FONT_W * 4;   /* バー開始 X */
    for (int x = bx; x < bx + fill; x++) {
        for (int y = 1; y < 7; y++) {   /* y=1..6 (6px 高) */
            _osd_pset(x, y);
        }
    }
}

/* ============================================================
 * PAUSE 表示 (画面中央、500ms 点滅)
 * ============================================================ */
void osd_draw_pause(void)
{
    uint32_t now = (uint32_t)(xTaskGetTickCount() * portTICK_PERIOD_MS);
    /* 500ms 周期で点滅: 250ms ON / 250ms OFF */
    if ((now / (PAUSE_BLINK_MS / 2)) & 1U) {
        return;   /* OFF 期間 */
    }

    /* "PAUSE" を画面中央に描画 */
    /* 5文字 × 8px = 40px, 中央 X = (128-40)/2 = 44 */
    /* 中央 Y = (64-8)/2 = 28 */
    const char *txt = "PAUSE";
    int cx = (CFG_PHYS_W - 5 * FONT_W) / 2;
    int cy = (CFG_PHYS_H - FONT_ROWS)  / 2;
    for (const char *p = txt; *p; p++, cx += FONT_W) {
        _osd_char(cx, cy, *p);
    }
}
