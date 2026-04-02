/**
 * @file  bad_data.h
 * PLACEHOLDER - replace with output from:
 *   python3 Codec.py -t e -p ./frames -n frame_ -s 0001 -e XXXX -o out.bad
 *   python3 Codec.py -t c -i out.bad -H bad_data.h
 * then copy bad_data.h to include/
 */

 #ifndef BAD_DATA_H
 #define BAD_DATA_H
 #include <stdint.h>
 #define BAD_DATA_WIDTH   128U
 #define BAD_DATA_HEIGHT  64U
 #define BAD_DATA_FRAMES  1U
 #define BAD_DATA_SIZE    20UL 
 const uint8_t bad_data[] = {
  0x13, 0x00, 0x9A, 0x93, 0x42, 0x61, 0x64, 0x02, 0x02, 0x02, 0x00, 0x80, 0x00, 0x40, 0x00, 0x08,
    0x00, 0xAC, 0x19, 0xE9, 0x39, 0x3C, 0xBE, 0x28, 0x77, 0x50, 0x0C, 0x8E, 0x20, 0x86, 0x80, 0x03
}
