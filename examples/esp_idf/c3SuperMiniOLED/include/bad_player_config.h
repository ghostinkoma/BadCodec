/**
 * @file  bad_player_config.h
 * @brief BadCodecPlayer ハードウェア設定
 * @version v0.6.0  (Protocol 514, SPEC rev.19)
 *
 * ESP32-C3 Mini + SSD1306 128x64 OLED (I2C)
 * 論理描画エリア: 72x40 (物理画面中央に配置)
 *
 * 変更方法:
 *   GPIO や解像度をここで変更するだけで全ソースに反映される。
 */

#ifndef BAD_PLAYER_CONFIG_H
#define BAD_PLAYER_CONFIG_H

#include <stdint.h>

/* ============================================================
 * I2C 設定
 * ============================================================ */
#define BAD_I2C_PORT        I2C_NUM_0
#define BAD_I2C_SDA         5           /* GPIO5 */
#define BAD_I2C_SCL         6           /* GPIO6 */
#define BAD_I2C_FREQ_HZ     1000000     /* 1 MHz */

/* ============================================================
 * SSD1306 設定
 * ============================================================ */
#define BAD_OLED_ADDR       0x3C        /* I2C アドレス */
#define BAD_OLED_WIDTH      128         /* 物理解像度 横 */
#define BAD_OLED_HEIGHT     64          /* 物理解像度 縦 */
#define BAD_OLED_PAGES      8           /* 64 / 8 */

/* ============================================================
 * BadCodec 動画設定
 *
 * エンコード時に指定した解像度と一致させること。
 *   python3 tools/Codec.py -t e ... → 72x40 でエンコード
 *   python3 tools/Codec.py -t c -i out.bad -H bad_data.h
 * ============================================================ */
#define BAD_VIDEO_WIDTH     72          /* エンコード解像度 横 */
#define BAD_VIDEO_HEIGHT    40          /* エンコード解像度 縦 */

/* 物理画面中央に配置するオフセット
 *   xOffset = (BAD_OLED_WIDTH  - BAD_VIDEO_WIDTH)  / 2
 *   yOffset = (BAD_OLED_HEIGHT - BAD_VIDEO_HEIGHT) / 2
 * ただし SSD1306 の物理列は 132 列 (表示は 128 列)
 *   xOffset = (132 - BAD_VIDEO_WIDTH) / 2 = 30
 * 参考コードと一致させる場合: xOffset=27, yOffset=23
 */
#define BAD_X_OFFSET        27          /* (132 - 72) / 2 */
#define BAD_Y_OFFSET        12          /* (40  - 40) / 2 = 0 → 中央寄せ */

/* ============================================================
 * フレームレート設定
 * ============================================================ */
#define BAD_FRAME_MS        29          /* 約 34fps (元動画に合わせて調整) */

/* ============================================================
 * 読み込み元の選択
 * BAD_SOURCE_FLASH : bad_data.h の埋め込み配列から読む
 * BAD_SOURCE_SD    : SD カードの /sdcard/output.bad から読む
 * どちらか一方だけ定義する
 * ============================================================ */
#define BAD_SOURCE_FLASH    1
/* #define BAD_SOURCE_SD    1 */

#ifdef BAD_SOURCE_SD
  #define BAD_SD_FILE       "/sdcard/output.bad"
  /* SD SPI 接続ピン (使用時は変更) */
  #define BAD_SD_CS         10
  #define BAD_SD_MOSI       7
  #define BAD_SD_MISO       2
  #define BAD_SD_CLK        3
#endif

#endif /* BAD_PLAYER_CONFIG_H */
