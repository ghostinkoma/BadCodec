# BadCodec

**Binary monochrome video codec for 8-bit microcontrollers**

[![Version](https://img.shields.io/badge/version-0.5.5-blue)](CHANGELOG.md)
[![Protocol](https://img.shields.io/badge/protocol-055-lightgrey)](SPEC.md)
[![License](https://img.shields.io/badge/license-Non--Commercial-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-pre--release-orange)]()

> **Pre-release**: We will maintain version 0.x.x until encoding/decoding consistency testing is complete.

---

## Overview

BadCodec is a codec designed for real-time playback of monochrome binary video on ultra-low-spec microcontrollers such as the LGT8F328 (2KB RAM) and microcontrollers like the ESP32.

**Design Philosophy:**

```
- 1-byte instruction set   Instruction determined immediately with a single 1-byte opcode
- Only one previous frame buffer   No dynamic memory allocation
- Bitwise operations only   No multiplication, division, or floating-point operations
- Fletcher-16 checksum   Addition only; 2B RAM
- read callback   Abstracts Flash/SD/SPIFFS/LittleFS
```

---

## Actual Compression Ratio

```
Example:
Bad Apple!! 128x64 / 6572 frames / 2-value monochrome / Lossless

  Uncompressed     : 6.42 MB  (100%)
  BadCodec   : 0.93 MB  (14.4%)  6.92x

Constraints:
  No encoding (no Huffman, arithmetic coding, etc.)
  No dictionary compression (no LZ77, LZW, etc.)
  Must be decodable in real time on an LGT8F328 (2KB RAM)

```

---

## Target Platforms

| Target | CPU | RAM | Recommended Resolution |
|-----------|-----|-----|----------|
| LGT8F328 (Minimum) | AVR-compatible 32MHz | 2KB | 128×64 |
| ESP32-Cx (Standard) | RISC-V 160MHz | 400KB | 320×240 or less |
| RP2350 (Maximum) | Cortex-M33 150MHz | 520KB | 640×480 or less |

---

## Instruction Set (Protocol 514)

### Frame-Level Instructions

| Instruction | Opcode | Description |
|------|-----------|------|
| RLE_FRAME | 0x30-0x37 | RLE for entire frame (8-scan pattern) |
| SKIP_FRAME | 0x39 | Maintain the previous frame as-is |
| FRAME_FILL_BLACK | 0x3A | Fill all pixels with black |
| FRAME_FILL_WHITE | 0x3B | Fill all pixels with white |
| BLOCK_STREAM | 0x3C | Block command stream |
| DELTA_FRAME | 0x3D | XOR difference frame RLE, 8-scan pattern |
| MASTER_FRAME | 0x3E | Raw bit data |
| INVERT_PREV_FRAME | 0x3F | Invert all bits of the previous frame |

### Block-Level Commands

| Command | Opcode | Size | Description |
|------|-----------|--------|------|
| SKIP_BLOCK | 0x80-0xBF | 1B | Copy previous frame (1-64 blocks) |
| FILL_BLOCK | 0x30-0x37 | 1B | Solid-color fill (1-4 blocks) |
| BLOCK_INVERT | 0x00-0x1F | 1B | Invert previous frame (1-32 blocks) |
| SHIFT_BIT | 0x40-0x7F | 1B | ±3-pixel fine shift difference |
| RLE_BLOCK_4 | 0x20-0x2F | 4B | 8-direction RLE, 4 runs |
| RLE_BLOCK_8 | 0x38-0x3B,0x3D-0x3E | 7B | Extended RLE, 8 runs |
| XOR_BLOCK | 0x3F | 2+NB | Prev XOR difference RLE |
| MASTER_BLOCK | 0x3C | 9B | Raw bit data |
| FOR | 0xC0-0xFF | 2B | Repeat the next instruction 4–65 times (minimum repeat=4) |

---

## Installation

```bash
git clone https://github.com/ghostinkoma/BadCodec.git
cd BadCodec
pip install Pillow numpy
```

---

## Usage

### Encoding

```bash
python3 tools/Codec.py -t e \
  -p ./frames \
  -n frame_ \
  -s 0001 \
  -e 6572 \
  -o output.bad
```

### Decoding

```bash
python3 tools/Codec.py -t d \
  -i output.bad \
  -p ./out \
  -n frame_ \
  -s 0001
```
### Options

| Option | Default | Description |
|-----------|-----------|------|
| `-t` | Required | `e`=Encode / `d`=Decode  / `c`= c header output |
| `-p` | Required | BMP directory |
| `-n` | `frame_` | File name prefix |
| `-s` | `0001` | Start frame number |
| `-e` | Required (encode) | End frame number |
| `-o` | `output.bad` | Output file name |
| `-i` | Required (decode) | Input .bad file |
| `-H` | Out put c header file name also bad_data | Input .bad file |

---

## Decoder for Microcontrollers

Use `tools/bad_decode.h` / `tools/bad_decode.cpp`.

```c
#include “bad_decode.h”

static uint8_t   gram[BAD_GRAM_SIZE(128, 64)];
static uint8_t   prev[BAD_GRAM_SIZE(128, 64)];
static bad_ctx_t ctx;

uint16_t my_read(bad_addr_t off, uint8_t *buf, uint16_t len) {
    for (uint16_t i = 0; i < len; i++)
        buf[i] = pgm_read_byte(VIDEO_DATA + off + i);
    return len;
}

void setup() {
    ctx.read     = my_read;
    ctx.gram     = gram;
    ctx.prev     = prev;
    ctx.buf_size = sizeof(gram);
    bad_init(&ctx);
}

void loop() {
    bad_result_t r = bad_next_frame(&ctx);
    if (r == BAD_OK)  display_write(ctx.gram);
    if (r == BAD_EOF) bad_rewind(&ctx);
}
```

---

## File Format

```
[19-byte header]
  2B: Header size (fixed at 19)
  2B: Fletcher-16 checksum
  3B: Magic number “Bad”
  2B: Protocol version (514)
  2B: Number of colors (2)
  2B: Image width
  2B: Image height
  2B: Block size (8)
  2B: Total number of frames

[Frame data]
  Repetition of FRAME_DELIMITER(0x38) + Frame Command
```

For details, see [SPEC.md](SPEC.md).

---

## Development Status

```
[Done] Codec specification finalized (SPEC.md rev.18)
[Done] Python encoder/decoder (Protocol 055)
[Done] Multi-CPU parallel encoding
[Done] Self-Verify mechanism
[Done] C decoder (bad_decode.h / bad_decode.cpp)
[Done] FOR optimization: Recursive optimization calculation (rev.18)
[Done] DELTA_FRAME (rev.16)
[Done] C decoder hardware verification

## License

Free for non-commercial use. Please contact us for commercial use.

**Contact:** ghostinkoma@gmail.com 
