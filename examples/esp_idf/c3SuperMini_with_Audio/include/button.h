/**
 * @file  button.h
 * @brief ボタン入力モジュール
 * @version v1.0.0
 *
 * ボタン定義:
 *   BTN_PLAY   停止/再生トグル            チャタリング 150ms
 *   BTN_OSD    OSDオーバーレイ ON/OFFトグル  3bit操作  チャタリング 150ms
 *   BTN_VOL_UP 音量+  16段階              チャタリング 80ms
 *   BTN_VOL_DN 音量−  16段階              チャタリング 80ms
 *
 * GPIO: 内部プルアップ使用 (外部 10kΩ プルアップ不要)
 *   押下=LOW (アクティブ LOW)
 *
 * ボタンが未接続でも動作する (常に OPEN=HIGH=非押下扱い)
 */

#ifndef BUTTON_H
#define BUTTON_H

#include <stdint.h>
#include "config.h"

/* ---- ボタン GPIO 割り当て (config.h で上書き可) ----------- */
#ifndef CFG_BTN_PLAY
#define CFG_BTN_PLAY    4   /* 停止/再生 */
#endif
#ifndef CFG_BTN_OSD
#define CFG_BTN_OSD     3   /* OSD ON/OFF */
#endif
#ifndef CFG_BTN_VOL_UP
#define CFG_BTN_VOL_UP  1   /* 音量+ */
#endif
#ifndef CFG_BTN_VOL_DN
#define CFG_BTN_VOL_DN  2   /* 音量− */
#endif

/* ---- チャタリング除去時間 --------------------------------- */
#define BTN_DEBOUNCE_PLAY_MS   150U
#define BTN_DEBOUNCE_OSD_MS    150U
#define BTN_DEBOUNCE_VOL_MS     80U

/* ---- 音量設定 --------------------------------------------- */
#define BTN_VOL_STEPS   16U    /* 0〜15 の 16段階 */
#define BTN_VOL_SHOW_MS 1500U  /* 音量表示時間 ms */

/* ---- 公開状態 --------------------------------------------- */
typedef struct {
    uint8_t  paused;       /* 1=停止中 */
    uint8_t  osd_mask;     /* bit0=CPU bit1=FPS bit2=VU */
    uint8_t  vol_step;     /* 0〜15 */
    uint8_t  vol_show;     /* 1=音量表示中 */
    uint32_t vol_show_end; /* 表示終了時刻 (ms) */
} btn_state_t;

extern btn_state_t g_btn;

/**
 * ボタンモジュール初期化。
 * GPIO を内部プルアップ入力として設定し、定期チェックタスクを起動する。
 *
 * OSD 初期状態は config.h の CFG_OSD_CPU/FPS/VU から自動生成する。
 *   CFG_OSD_CPU=1, CFG_OSD_FPS=1, CFG_OSD_VU=0 の場合:
 *   g_btn.osd_mask = 0x03 (CPU+FPS のみ表示)
 *
 * vol_default: config.h の CFG_AUDIO_VOL をステップ値に変換して使用。
 */
void button_init(uint8_t vol_default_step);

/**
 * 現在の音量ステップから実際の VOL 値 (0-256) を返す。
 * adpcm_set_vol() に渡すか、動的 VOL として使用する。
 */
uint16_t button_get_vol(void);

#endif /* BUTTON_H */
