/**
 * @file  bad_data.h
 * @brief BadCodec 動画データ (プレースホルダー)
 *
 * このファイルは Codec.py -t c で生成してください:
 *   python3 tools/Codec.py -t c -i output.bad -H bad_data.h
 *
 * 生成したファイルをこのディレクトリ (include/) に配置してください。
 */

#ifndef BAD_DATA_H
#define BAD_DATA_H

#include <stdint.h>

#define BAD_DATA_WIDTH    72U
#define BAD_DATA_HEIGHT   40U
#define BAD_DATA_FRAMES   1U
#define BAD_DATA_SIZE     20UL

/* ダミーデータ: 実際の動画データに差し替えること */
#ifdef __AVR__
#include <avr/pgmspace.h>
const uint8_t bad_data[] PROGMEM = {
#else
const uint8_t bad_data[] = {
#endif
    /* BadCodec ヘッダ (19B) + SKIP_FRAME (1B) のダミー */
    0x13, 0x00, 0x00, 0x00, 0x42, 0x61, 0x64, 0x02,
    0x02, 0x02, 0x00, 0x48, 0x00, 0x28, 0x00, 0x08,
    0x00, 0x01, 0x00, 0x39
};

#endif /* BAD_DATA_H */
