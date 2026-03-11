#!/usr/bin/env python3
“””
BadCodec v0.5.1
Based on BadCodec 完全実装仕様書 rev.11

- MD5 -> Fletcher-16 に変更（マイコン向け最低解）

Usage:
Encode: python badcodec.py -t e -p ./frames -n frame_ -s 0001 -e 6572 -o output.bad
Decode: python badcodec.py -t d -i output.bad -p ./out  -n frame_ -s 0001
“””

import os
import sys
import time
import argparse
import struct
import numpy as np
from PIL import Image
from collections import deque
from multiprocessing import Pool, cpu_count

# ============================================================

# ANSI Colors

# ============================================================

C_RESET   = “\033[0m”
C_BOLD    = “\033[1m”
C_GREEN   = “\033[92m”   # S: Skip
C_WHITE   = “\033[97m”   # F: Fill
C_YELLOW  = “\033[33m”   # R: RLE
C_CYAN    = “\033[36m”   # X: Shift
C_RED     = “\033[91m”   # M: Master
C_MAGENTA = “\033[35m”   # I: Invert

TYPE_COLOR = {
‘S’: C_GREEN, ‘F’: C_WHITE, ‘R’: C_YELLOW,
‘X’: C_CYAN,  ‘M’: C_RED,   ‘I’: C_MAGENTA,
}

# ============================================================

# Opcodes  (仕様書 第5章 bit tree より)

# ============================================================

# — Block-level —

# 0x00-0x1F : BLOCK_INVERT   (000nnnnn)

# 0x20-0x2F : RLE_BLOCK      (0010pppc)  pp=scan_pattern bit3-1, c=start_color

# 0x30-0x37 : FILL_BLOCK     (00110cnn)  c=color, nn=count-1

# 0x38-0x3F : FRAME_CONTROL  (00111xxx)  ※ block context では MASTER_BLOCK=0x3C のみ使用

# 0x40-0x7F : SHIFT_BIT      (01sxxcyy)  s=sign_x, xx=mag_x, c=sign_y, yy=mag_y

# 0x80-0xBF : SKIP_BLOCK     (10nnnnnn)

# 0xC0-0xFF : FOR            (11nnnnnn)

# — Frame-level (FRAME_CONTROL / special) —

OP_BLOCK_STREAM     = 0x30   # ※ frame context では BLOCK_STREAM として扱う
OP_FRAME_DELIMITER  = 0x38
OP_SKIP_FRAME       = 0x39
OP_FRAME_FILL_BLACK = 0x3A
OP_FRAME_FILL_WHITE = 0x3B
OP_MASTER_BLOCK_F   = 0x3C
OP_RLE_FRAME        = 0x3D
OP_MASTER_FRAME     = 0x3E
OP_INVERT_PREV      = 0x3F
OP_EXT_PREFIX       = 0xFF   # 拡張命令プレフィックス

# Block-level MASTER_BLOCK opcode (FRAME_CONTROL range 内)

OP_MASTER_BLOCK_B   = 0x3C

# ============================================================

# Constants

# ============================================================

BLOCK_SIZE = 8
VERSION    = 510   # protocol version (file format)

# ============================================================

# Fletcher-16  (仕様書 3-3節)

# マイコン向け最低解チェックサム

# s1 = Σ data[i]      (mod 256)

# s2 = Σ s1           (mod 256)

# 格納値 = (s2 << 8) | s1  リトルエンディアン2バイト

# 

# コード量 : 数十バイト（MD5の1/20）

# RAM      : 2バイト（MD5の1/32）

# テーブル : 不要

# 8bit演算 : 加算のみで完結

# ============================================================

def fletcher16(data: bytes) -> int:
s1, s2 = 0, 0
for b in data:
s1 = (s1 + b) & 0xFF
s2 = (s2 + s1) & 0xFF
return (s2 << 8) | s1

# ============================================================

# Header  (仕様書 第3章)

# [2: hdr_size] [2: Fletcher-16] [3: “Bad”] [2: ver] [2: colors]

# [2: w] [2: h] [2: blksize] [2: total_frames]

# ヘッダー全体サイズ: 2 + 2 + 15 = 19バイト固定

# ============================================================

def encode_header(w, h, total_frames):
body = bytearray()
body.extend(b’Bad’)
body.extend(struct.pack(’<HHHHHH’, VERSION, 2, w, h, BLOCK_SIZE, total_frames))
chk  = fletcher16(bytes(body))
size = 2 + 2 + len(body)   # hdr_size(2) + Fletcher-16(2) + body(15)
out  = bytearray()
out.extend(struct.pack(’<H’, size))
out.extend(struct.pack(’<H’, chk))
out.extend(body)
return bytes(out)

def decode_header(data):
“”“Returns (w, h, block_size, total_frames, hdr_size) or raises ValueError.”””
if len(data) < 4:
raise ValueError(“File too short for header.”)
hdr_size   = struct.unpack(’<H’, data[0:2])[0]
stored_chk = struct.unpack(’<H’, data[2:4])[0]
body       = data[4:hdr_size]
calc_chk   = fletcher16(bytes(body))
if stored_chk != calc_chk:
raise ValueError(
f”Fletcher-16 Mismatch! “
f”stored=0x{stored_chk:04X} calc=0x{calc_chk:04X} “
f”File may be corrupted.”)
if body[:3] != b’Bad’:
raise ValueError(f”Invalid magic number: {body[:3]}”)
_ver, _colors, w, h, blk, total_f = struct.unpack(’<HHHHHH’, body[3:15])
return w, h, blk, total_f, hdr_size

# ============================================================

# Scan paths for RLE_BLOCK  (仕様書 6-5節)

# Pattern index  = (scan_dir << 2) | start_pos_bits

# scan_dir   : 0=horizontal, 1=vertical

# start_pos  : bit1=start_y(0=top,1=bottom), bit0=start_x(0=left,1=right)

# ============================================================

def _build_scan_path(p_idx):
scan_dir  = (p_idx >> 2) & 1
sp        = p_idx & 3
start_x   = 7 if (sp & 1) else 0
start_y   = 7 if (sp & 2) else 0
step_x    = -1 if (sp & 1) else 1
step_y    = -1 if (sp & 2) else 1
coords = []
if scan_dir == 0:
y = start_y
while 0 <= y < 8:
x = start_x
while 0 <= x < 8:
coords.append((x, y))
x += step_x
y += step_y
else:
x = start_x
while 0 <= x < 8:
y = start_y
while 0 <= y < 8:
coords.append((x, y))
y += step_y
x += step_x
return np.array(coords, dtype=np.int32)

SCAN_PATHS = [_build_scan_path(i) for i in range(8)]

# ============================================================

# RLE bit-pack / unpack  (仕様書 6-5節)

# 6bit × 4 runs → 3 bytes リトルエンディアン

# ============================================================

def pack_rle(runs):
r = (list(runs) + [0, 0, 0, 0])[:4]
b0, b1, b2, b3 = [int(x) & 0x3F for x in r]
byte0 = b0 | ((b1 & 0x03) << 6)
byte1 = ((b1 >> 2) & 0x0F) | ((b2 & 0x0F) << 4)
byte2 = ((b2 >> 4) & 0x03) | (b3 << 2)
return bytes([byte0, byte1, byte2])

def unpack_rle(data):
b0, b1, b2 = data[0], data[1], data[2]
r0 = b0 & 0x3F
r1 = ((b0 >> 6) & 0x03) | ((b1 & 0x0F) << 2)
r2 = ((b1 >> 4) & 0x0F) | ((b2 & 0x03) << 4)
r3 = (b2 >> 2) & 0x3F
return [r0, r1, r2, r3]

# ============================================================

# SHIFT_BIT helpers  (仕様書 6-3節)

# ============================================================

def apply_shift(block, sx, sy):
“””
Shift 8x8 block by (sx, sy).
Padding: edge pixel value of the pre-shift block.
+X = right, -X = left, +Y = down, -Y = up
“””
result = block.astype(np.uint8).copy()
# X shift
if sx > 0:
for _ in range(sx):
edge = result[:, -1:]
result = np.hstack([edge, result[:, :-1]])
elif sx < 0:
for _ in range(-sx):
edge = result[:, :1]
result = np.hstack([result[:, 1:], edge])
# Y shift
if sy > 0:
for _ in range(sy):
edge = result[-1:, :]
result = np.vstack([edge, result[:-1, :]])
elif sy < 0:
for _ in range(-sy):
edge = result[:1, :]
result = np.vstack([result[1:, :], edge])
return result

# ============================================================

# Block encoder  (仕様書 9-1節)

# ============================================================

def _try_rle(block):
“”“Try all 8 scan patterns × 2 start colors. Return (opcode, 3bytes) or None.”””
best = None
for p_idx in range(8):
path   = SCAN_PATHS[p_idx]
pixels = block[path[:, 1], path[:, 0]]
for start_col in range(2):
runs  = []
curr  = start_col
count = 0
valid = True
for px in pixels:
if int(px) == curr:
count += 1
else:
if count > 63:
valid = False; break
runs.append(count)
curr  = 1 - curr
count = 1
if not valid:
continue
if count > 63:
continue
runs.append(count)
if len(runs) > 4:
continue
opcode = 0x20 | (p_idx << 1) | start_col
if best is None:
best = (opcode, pack_rle(runs))
return best

def _encode_master_block(block):
raw = np.packbits(block.flatten())
return bytes([OP_MASTER_BLOCK_B]) + bytes(raw)  # 9 bytes total

def encode_block(curr_b, prev_b):
“””
Encode one 8×8 block. Returns (bytes, type_char).
All candidates are evaluated; smallest wins.
“””
candidates = []

```
# 1. FILL_BLOCK  (1 byte)
if np.all(curr_b == 0):
    candidates.append((bytes([0x30]), 'F'))   # FILL_BLACK ×1
if np.all(curr_b == 1):
    candidates.append((bytes([0x34]), 'F'))   # FILL_WHITE ×1

# 2. SKIP_BLOCK  (1 byte)
if np.array_equal(curr_b, prev_b):
    candidates.append((bytes([0x80]), 'S'))

# 3. BLOCK_INVERT  (1 byte)
if np.array_equal(curr_b, 1 - prev_b):
    candidates.append((bytes([0x00]), 'I'))

# 4. SHIFT_BIT  (1 byte) — try all ±3 combinations
for sx in range(-3, 4):
    for sy in range(-3, 4):
        if sx == 0 and sy == 0:
            continue
        if np.array_equal(apply_shift(prev_b, sx, sy), curr_b):
            sign_x = 1 if sx < 0 else 0
            mag_x  = abs(sx)
            sign_y = 1 if sy < 0 else 0
            mag_y  = abs(sy)
            op = 0x40 | (sign_x << 5) | (mag_x << 3) | (sign_y << 2) | mag_y
            candidates.append((bytes([op]), 'X'))
            break
    else:
        continue
    break

# 5. RLE_BLOCK  (4 bytes: opcode + 3)
rle = _try_rle(curr_b)
if rle is not None:
    candidates.append((bytes([rle[0]]) + rle[1], 'R'))

# 6. MASTER_BLOCK  (9 bytes: opcode + 8) — always available
candidates.append((_encode_master_block(curr_b), 'M'))

return min(candidates, key=lambda c: len(c[0]))
```

# ============================================================

# Multi-block merge  (SKIP / INVERT / FILL 連続をまとめる)

# ============================================================

def _merge_multiblock(raw_cmds, raw_types):
result_c, result_t = [], []
i = 0
while i < len(raw_cmds):
t = raw_types[i]
c = raw_cmds[i]

```
    if t == 'S' and c == bytes([0x80]):
        j = i
        while j < len(raw_cmds) and raw_types[j] == 'S' \
                and raw_cmds[j] == bytes([0x80]):
            j += 1
        rem = j - i
        while rem > 0:
            n = min(rem, 64)
            result_c.append(bytes([0x80 | (n - 1)]))
            result_t.append('S')
            rem -= n
        i = j

    elif t == 'I' and c == bytes([0x00]):
        j = i
        while j < len(raw_cmds) and raw_types[j] == 'I' \
                and raw_cmds[j] == bytes([0x00]):
            j += 1
        rem = j - i
        while rem > 0:
            n = min(rem, 32)
            result_c.append(bytes([0x00 | (n - 1)]))
            result_t.append('I')
            rem -= n
        i = j

    elif t == 'F' and len(c) == 1:
        base = c[0] & 0xFC   # mask off count bits
        j    = i
        while j < len(raw_cmds) and raw_types[j] == 'F' \
                and len(raw_cmds[j]) == 1 \
                and (raw_cmds[j][0] & 0xFC) == base:
            j += 1
        rem = j - i
        while rem > 0:
            n = min(rem, 4)
            result_c.append(bytes([base | (n - 1)]))
            result_t.append('F')
            rem -= n
        i = j

    else:
        result_c.append(c)
        result_t.append(t)
        i += 1

return result_c, result_t
```

# ============================================================

# FOR optimizer  (仕様書 9-2節)

# ============================================================

def optimize_for(cmds, types):
“””
Merge consecutive identical single-byte commands using FOR.
FOR 0 (2×) and FOR 1 (3×) are forbidden → write literally.
“””
out_b = bytearray()
out_t = []
i     = 0
while i < len(cmds):
cmd = cmds[i]
t   = types[i]
if len(cmd) != 1:
out_b.extend(cmd)
out_t.append(t)
i += 1
continue
j = i + 1
while j < len(cmds) and cmds[j] == cmd and len(cmds[j]) == 1:
j += 1
n = j - i
rem = n
while rem > 0:
if rem <= 3:
for _ in range(rem):
out_b.extend(cmd)
out_t.append(t)
rem = 0
else:
take = min(rem, 65)
for_byte = 0xC0 | (take - 2)
out_b.extend([for_byte, cmd[0]])
out_t.append(t)
rem -= take
i = j
return bytes(out_b), out_t

# ============================================================

# RLE_FRAME encoder  (仕様書 第7章)

# ============================================================

def encode_rle_frame(frame_2d):
pixels = frame_2d.flatten().tolist()
out    = bytearray()
i = 0
while i < len(pixels):
color = pixels[i]
count = 1
while i + count < len(pixels) and pixels[i + count] == color and count < 127:
count += 1
out.append((color << 7) | count)
i += count
return bytes(out)

# ============================================================

# Frame-level decode  (仕様書 第7章)

# ============================================================

def decode_frame(data, prev_f, w, h):
bx     = w // BLOCK_SIZE
by_cnt = h // BLOCK_SIZE
n_blk  = bx * by_cnt
op     = data[0]

```
if op == OP_SKIP_FRAME:
    return prev_f.copy()
if op == OP_FRAME_FILL_BLACK:
    return np.zeros((h, w), dtype=np.uint8)
if op == OP_FRAME_FILL_WHITE:
    return np.ones((h, w), dtype=np.uint8)
if op == OP_INVERT_PREV:
    return (1 - prev_f).astype(np.uint8)
if op == OP_MASTER_FRAME:
    bits = np.unpackbits(np.frombuffer(data[1:], dtype=np.uint8))[:w * h]
    return bits.reshape(h, w).astype(np.uint8)
if op == OP_RLE_FRAME:
    return _decode_rle_frame(data[1:], w, h)
if op == OP_BLOCK_STREAM:
    return _decode_block_stream(data[1:], prev_f, w, h, bx, n_blk)

raise ValueError(f"Unknown frame opcode: 0x{op:02X}")
```

def _decode_rle_frame(rle, w, h):
total  = w * h
pixels = []
for byte in rle:
color  = (byte >> 7) & 1
length = byte & 0x7F
if length == 0:
break
pixels.extend([color] * length)
if len(pixels) >= total:
break
arr = np.array(pixels[:total], dtype=np.uint8)
return arr.reshape(h, w)

def _decode_block_stream(stream, prev_f, w, h, bx, n_blk):
curr_f = np.zeros((h, w), dtype=np.uint8)
ptr    = 0
b_idx  = 0

```
def do_block(cmd, b_i, p):
    nonlocal curr_f
    by_i = b_i // bx
    bx_i = b_i % bx
    y    = by_i * BLOCK_SIZE
    x    = bx_i * BLOCK_SIZE

    # SKIP_BLOCK
    if cmd & 0x80:
        cnt = (cmd & 0x3F) + 1
        for k in range(cnt):
            if b_i + k < n_blk:
                yy = ((b_i + k) // bx) * BLOCK_SIZE
                xx = ((b_i + k) % bx) * BLOCK_SIZE
                curr_f[yy:yy+BLOCK_SIZE, xx:xx+BLOCK_SIZE] = \
                    prev_f[yy:yy+BLOCK_SIZE, xx:xx+BLOCK_SIZE]
        return b_i + cnt, p

    # BLOCK_INVERT
    if cmd <= 0x1F:
        cnt = (cmd & 0x1F) + 1
        for k in range(cnt):
            if b_i + k < n_blk:
                yy = ((b_i + k) // bx) * BLOCK_SIZE
                xx = ((b_i + k) % bx) * BLOCK_SIZE
                curr_f[yy:yy+BLOCK_SIZE, xx:xx+BLOCK_SIZE] = \
                    1 - prev_f[yy:yy+BLOCK_SIZE, xx:xx+BLOCK_SIZE]
        return b_i + cnt, p

    # RLE_BLOCK
    if 0x20 <= cmd <= 0x2F:
        p_idx     = (cmd >> 1) & 0x07
        start_col = cmd & 0x01
        runs  = unpack_rle(stream[p:p+3])
        path  = SCAN_PATHS[p_idx]
        pix   = []
        col   = start_col
        for r in runs:
            pix.extend([col] * r)
            col = 1 - col
        blk = np.zeros(64, dtype=np.uint8)
        blk[:min(64, len(pix))] = pix[:64]
        curr_f[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE] = blk.reshape(BLOCK_SIZE, BLOCK_SIZE)
        return b_i + 1, p + 3

    # FILL_BLOCK
    if 0x30 <= cmd <= 0x37:
        color = (cmd >> 2) & 1
        cnt   = (cmd & 0x03) + 1
        for k in range(cnt):
            if b_i + k < n_blk:
                yy = ((b_i + k) // bx) * BLOCK_SIZE
                xx = ((b_i + k) % bx) * BLOCK_SIZE
                curr_f[yy:yy+BLOCK_SIZE, xx:xx+BLOCK_SIZE] = color
        return b_i + cnt, p

    # MASTER_BLOCK
    if cmd == OP_MASTER_BLOCK_B:
        bits = np.unpackbits(np.frombuffer(stream[p:p+8], dtype=np.uint8))
        curr_f[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE] = \
            bits.reshape(BLOCK_SIZE, BLOCK_SIZE)
        return b_i + 1, p + 8

    # SHIFT_BIT
    if 0x40 <= cmd <= 0x7F:
        sign_x = (cmd >> 5) & 1
        mag_x  = (cmd >> 3) & 3
        sign_y = (cmd >> 2) & 1
        mag_y  =  cmd       & 3
        sx     = -mag_x if sign_x else mag_x
        sy     = -mag_y if sign_y else mag_y
        pb     = prev_f[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE]
        curr_f[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE] = apply_shift(pb, sx, sy)
        return b_i + 1, p

    # Unknown — skip 1 block
    return b_i + 1, p

while b_idx < n_blk and ptr < len(stream):
    cmd = stream[ptr]; ptr += 1
    # FOR
    if cmd & 0xC0 == 0xC0:
        repeat = (cmd & 0x3F) + 2
        if ptr >= len(stream):
            break
        ncmd = stream[ptr]; ptr += 1
        for _ in range(repeat):
            if b_idx >= n_blk:
                break
            b_idx, ptr = do_block(ncmd, b_idx, ptr)
        continue
    b_idx, ptr = do_block(cmd, b_idx, ptr)

return curr_f
```

# ============================================================

# Frame size calculator  (stream parser に必要)

# ============================================================

def frame_data_size(data, offset, w, h):
“”“Return byte size of one frame starting at data[offset].”””
op = data[offset]
# Fixed 1-byte frames
if op in (OP_SKIP_FRAME, OP_FRAME_FILL_BLACK,
OP_FRAME_FILL_WHITE, OP_INVERT_PREV):
return 1
# MASTER_FRAME: 1 + w*h/8
if op == OP_MASTER_FRAME:
return 1 + (w * h) // 8
# RLE_FRAME: read until pixel count reached
if op == OP_RLE_FRAME:
total  = w * h
count  = 0
i      = offset + 1
while count < total and i < len(data):
byte   = data[i]
length = byte & 0x7F
if length == 0:
i += 1; break
count += length
i     += 1
return i - offset
# BLOCK_STREAM: parse block commands until all blocks consumed
if op == OP_BLOCK_STREAM:
bx     = w // BLOCK_SIZE
n_blk  = bx * (h // BLOCK_SIZE)
b_idx  = 0
ptr    = offset + 1
while b_idx < n_blk and ptr < len(data):
cmd = data[ptr]; ptr += 1
# FOR
if cmd & 0xC0 == 0xC0:
repeat = (cmd & 0x3F) + 2
if ptr >= len(data):
break
ncmd = data[ptr]; ptr += 1
advance = _block_advance(ncmd)
extra   = _block_extra_bytes(ncmd)
for _ in range(repeat):
if b_idx >= n_blk:
break
b_idx += advance
ptr   += extra
continue
b_idx += _block_advance(cmd)
ptr   += _block_extra_bytes(cmd)
return ptr - offset
return 1

def _block_advance(cmd):
“”“How many blocks does this command consume?”””
if cmd & 0x80:          return (cmd & 0x3F) + 1   # SKIP_BLOCK
if cmd <= 0x1F:         return (cmd & 0x1F) + 1   # BLOCK_INVERT
if 0x20 <= cmd <= 0x2F: return 1                   # RLE_BLOCK
if 0x30 <= cmd <= 0x37: return (cmd & 0x03) + 1   # FILL_BLOCK
if cmd == 0x3C:         return 1                   # MASTER_BLOCK
if 0x40 <= cmd <= 0x7F: return 1                   # SHIFT_BIT
return 1

def _block_extra_bytes(cmd):
“”“How many extra data bytes follow this command?”””
if 0x20 <= cmd <= 0x2F: return 3   # RLE_BLOCK: 3 bytes
if cmd == 0x3C:         return 8   # MASTER_BLOCK: 8 bytes
return 0

# ============================================================

# Self-verify  (仕様書 13-5節)

# ============================================================

def _self_verify(frame_idx, data, types, original, prev_f, w, h):
err = None
try:
decoded = decode_frame(data, prev_f, w, h)
if np.array_equal(decoded, original):
return frame_idx, data, types, None
err = “verify mismatch”
except Exception as e:
err = str(e)

```
# Fallback: MASTER_FRAME
fb_data = bytes([OP_MASTER_FRAME]) + bytes(np.packbits(original.flatten()))
n_blk   = (w // BLOCK_SIZE) * (h // BLOCK_SIZE)
try:
    decoded2 = decode_frame(fb_data, prev_f, w, h)
    if np.array_equal(decoded2, original):
        return frame_idx, fb_data, ['M'] * n_blk, f"fallback({err})"
except Exception as e2:
    err = f"{err} | fallback_fail: {e2}"

return frame_idx, fb_data, ['M'] * n_blk, f"CRITICAL frame {frame_idx}: {err}"
```

# ============================================================

# Frame encoder worker  (multiprocessing)

# ============================================================

def encode_frame_worker(args):
frame_idx, curr_path, prev_path, w, h = args

```
curr_f = np.array(Image.open(curr_path).convert('1'), dtype=np.uint8)
if prev_path and os.path.exists(prev_path):
    prev_f = np.array(Image.open(prev_path).convert('1'), dtype=np.uint8)
else:
    prev_f = np.zeros((h, w), dtype=np.uint8)

bx     = w // BLOCK_SIZE
n_blk  = bx * (h // BLOCK_SIZE)
raw_sz = (w * h) // 8

# --- Frame-level single commands (候補D) ---
if np.array_equal(curr_f, prev_f):
    return _self_verify(frame_idx, bytes([OP_SKIP_FRAME]),
                        ['S'] * n_blk, curr_f, prev_f, w, h)
if np.all(curr_f == 0):
    return _self_verify(frame_idx, bytes([OP_FRAME_FILL_BLACK]),
                        ['F'] * n_blk, curr_f, prev_f, w, h)
if np.all(curr_f == 1):
    return _self_verify(frame_idx, bytes([OP_FRAME_FILL_WHITE]),
                        ['F'] * n_blk, curr_f, prev_f, w, h)
if np.array_equal(curr_f, 1 - prev_f):
    return _self_verify(frame_idx, bytes([OP_INVERT_PREV]),
                        ['I'] * n_blk, curr_f, prev_f, w, h)

# --- Block-level encoding ---
raw_cmds, raw_types = [], []
for b in range(n_blk):
    y = (b // bx) * BLOCK_SIZE
    x = (b %  bx) * BLOCK_SIZE
    cmd, t = encode_block(
        curr_f[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE],
        prev_f[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE])
    raw_cmds.append(cmd)
    raw_types.append(t)

merged_c, merged_t   = _merge_multiblock(raw_cmds, raw_types)
opt_bytes, opt_types = optimize_for(merged_c, merged_t)

# Candidate A: BLOCK_STREAM
cand_a = bytes([OP_BLOCK_STREAM]) + opt_bytes

# Candidate B: RLE_FRAME
rle_body = encode_rle_frame(curr_f)
cand_b   = bytes([OP_RLE_FRAME]) + rle_body

# Candidate C: MASTER_FRAME
cand_c = bytes([OP_MASTER_FRAME]) + bytes(np.packbits(curr_f.flatten()))

best = min(
    (cand_a, opt_types),
    (cand_b, ['R'] * n_blk),
    (cand_c, ['M'] * n_blk),
    key=lambda x: len(x[0])
)
data, types = best

return _self_verify(frame_idx, data, types, curr_f, prev_f, w, h)
```

# ============================================================

# UI helpers

# ============================================================

def _bar_hash(curr, total, width=40):
r    = curr / total if total > 0 else 0
done = int(width * r)
return f”[{’#’*done}{’.’*(width-done)}] {int(r*100):3d}% ({curr}/{total})”

def _bar_block(ratio, width=20):
done = int(width * ratio)
return f”{‘█’*done}{‘░’*(width-done)} {ratio*100:4.1f}%”

def _block_map(types, bx, max_rows=8, max_cols=30):
rows = min(max_rows, len(types) // bx if bx > 0 else 0)
cols = min(max_cols, bx)
lines = []
for r in range(rows):
row = “”
for c in range(cols):
idx = r * bx + c
t   = types[idx] if idx < len(types) else ‘?’
row += f”{TYPE_COLOR.get(t, C_RESET)}{t}{C_RESET} “
lines.append(row)
return lines

# ============================================================

# Frame-level FOR merge pass  (仕様書 9-2節・13章)

# 

# 判定ルール：

# フレーム命令を素直に書いて良いのは

# 「直前がフレーム命令でない」かつ「直前と同じフレーム命令でない」

# それ以外は FOR への包含を検討する

# 

# FOR 禁止：FOR 0（2回）・FOR 1（3回） → N≦3 は個別に書く

# FOR 上限：65回 → 65を超える場合は次のFORブロックに分割

# ============================================================

SINGLE_BYTE_FRAME_OPS = {
OP_SKIP_FRAME,
OP_FRAME_FILL_BLACK,
OP_FRAME_FILL_WHITE,
OP_INVERT_PREV,
}

def merge_frame_for(ordered_frames):
“””
ordered_frames: list of (data_bytes, blk_types)
全フレームを順番に受け取り、連続するフレーム命令を FOR でまとめる。

```
Returns: (merged_bytes, display_list)
  display_list: [(data_bytes, blk_types), ...] UIに渡す順序付きリスト
"""
out     = bytearray()
display = []   # UI表示用に (data, types) を保持
i       = 0
n       = len(ordered_frames)

while i < n:
    data, types = ordered_frames[i]

    # 単一バイトフレーム命令かどうか
    if len(data) == 1 and data[0] in SINGLE_BYTE_FRAME_OPS:
        op = data[0]

        # 連続する同じ命令を数える（FOR上限65まで）
        j = i + 1
        while (j < n
               and len(ordered_frames[j][0]) == 1
               and ordered_frames[j][0][0] == op
               and (j - i) < 65):
            j += 1
        count = j - i   # 連続数

        rem = count
        while rem > 0:
            take = min(rem, 65)
            if take <= 3:
                # FOR 0・1 禁止 → 個別に書く
                for k in range(take):
                    out.extend([op])
                    display.append((bytes([op]), types))
            else:
                # FOR (take-2) + op の2バイトで表現
                for_byte = 0xC0 | (take - 2)
                out.extend([for_byte, op])
                # display は FOR対象フレームをまとめて1エントリとして記録
                display.append((bytes([for_byte, op]), types))
            rem -= take
        i = j

    else:
        # BLOCK_STREAM / RLE_FRAME / MASTER_FRAME はそのまま
        out.extend(data)
        display.append((data, types))
        i += 1

return bytes(out), display
```

# ============================================================

# Encoder

# ============================================================

def do_encode(args):
st_i    = int(args.start)
ed_i    = int(args.end)
pad     = len(args.start)
total_f = ed_i - st_i + 1

```
first = os.path.join(args.path, f"{args.suffix}{st_i:0{pad}d}.bmp")
if not os.path.exists(first):
    print(f"Error: {first} not found."); sys.exit(1)

img = Image.open(first)
w, h = img.size
if w % BLOCK_SIZE or h % BLOCK_SIZE:
    print(f"Error: {w}x{h} is not divisible by {BLOCK_SIZE}."); sys.exit(1)

bx     = w // BLOCK_SIZE
raw_sz = (w * h) // 8
ncpu   = cpu_count()

# Build task list
tasks = []
for i in range(st_i, ed_i + 1):
    curr = os.path.join(args.path, f"{args.suffix}{i:0{pad}d}.bmp")
    prev = os.path.join(args.path, f"{args.suffix}{i-1:0{pad}d}.bmp") \
           if i > st_i else None
    tasks.append((i, curr, prev, w, h))

print(f"\033[2J\033[H", end="")
print(f"Starting Encode (Shift & FOR optimized)...")
time.sleep(0.2)
print(f"\033[2J\033[H", end="")

# -------------------------------------------------------
# Phase 1: 並列エンコード → 全フレームを順序付きで収集
#   書き込みはまだ行わない。
#   フレームFOR最適化は全フレーム確定後に行うため。
# -------------------------------------------------------
write_ptr     = st_i
buffer        = {}
ordered_frames = []   # (data, blk_types) in frame order
win_90        = deque(maxlen=90)
errors        = []

with Pool(ncpu) as pool:
    for result in pool.imap_unordered(encode_frame_worker, tasks):
        f_idx, enc_data, blk_types, err = result
        if err:
            errors.append(err)
        buffer[f_idx] = (enc_data, blk_types)

        # 順序保証バッファから連続して取り出す
        while write_ptr in buffer:
            bd, bt = buffer.pop(write_ptr)
            ordered_frames.append((bd, bt))

            ratio = len(bd) / raw_sz * 100
            win_90.append(ratio)
            avg   = sum(win_90) / len(win_90)
            curr_c = len(ordered_frames)
            p_rat  = curr_c / total_f

            sys.stdout.write('\033[H')
            print(f"{C_BOLD}BadCodec v0.5.1 Encoder{C_RESET}"
                  f"  [{ncpu} cores]  Phase 1/2: Encoding")
            print(_bar_block(p_rat))
            print(f"Frame: {write_ptr:04d} | "
                  f"Size: {len(bd):6d}B | "
                  f"Avg: {avg:5.2f}%")

            for line in _block_map(bt, bx):
                print(line)
            shown = len(_block_map(bt, bx))
            for _ in range(min(8, h // BLOCK_SIZE) - shown):
                print()

            write_ptr += 1

# -------------------------------------------------------
# Phase 2: フレームFOR最適化マージ
#   連続する同一フレーム命令を FOR に包含する。
#   この処理は全フレームが確定してからでないと正確に行えない。
# -------------------------------------------------------
sys.stdout.write('\033[H')
print(f"{C_BOLD}BadCodec v0.5.1 Encoder{C_RESET}"
      f"  [{ncpu} cores]  Phase 2/2: Frame FOR merge...")
print()

merged_stream, display_list = merge_frame_for(ordered_frames)

# -------------------------------------------------------
# Phase 3: ファイル書き込み
# -------------------------------------------------------
header      = encode_header(w, h, total_f)
total_bytes = len(header) + len(merged_stream)

with open(args.output, 'wb') as f:
    f.write(header)
    f.write(merged_stream)

raw_total = (w * h * total_f) // 8
print(f"{C_BOLD}Done.{C_RESET} Saved to {args.output} "
      f"({total_bytes:,} bytes, "
      f"{total_bytes / raw_total * 100:.1f}% of raw)")
print(f"Frame FOR merged stream: {len(merged_stream):,} bytes "
      f"(before merge: "
      f"{sum(len(d) for d,_ in ordered_frames):,} bytes)")

if errors:
    print(f"\n{C_RED}Warnings ({len(errors)}):{C_RESET}")
    for e in errors[:10]:
        print(f"  {e}")
```

# ============================================================

# Decoder

# ============================================================

def do_decode(args):
if not args.input:
print(“Error: -i is required for decode.”); sys.exit(1)
if not os.path.exists(args.input):
print(f”Error: {args.input} not found.”); sys.exit(1)

```
with open(args.input, 'rb') as f:
    raw = f.read()

try:
    w, h, blk, total_f, hdr_size = decode_header(raw)
except ValueError as e:
    print(f"Header error: {e}"); sys.exit(1)

st_i   = int(args.start)
pad    = len(args.start)
ed_lim = int(args.end) if args.end else st_i + total_f - 1
decode_count = min(total_f, ed_lim - st_i + 1)
ncpu = cpu_count()

os.makedirs(args.path, exist_ok=True)

print(f"\033[2J\033[H", end="")
print(f"{C_BOLD}BadCodec v0.5.1 Decoder{C_RESET}")
print(f"Input : {args.input}")
print(f"Output: {os.path.join(args.path, args.suffix)}%04d.bmp")
print(f"Frames: {total_f}  Size: {w}×{h}")
print(f"Starting Decode...")
time.sleep(0.3)
print(f"\033[2J\033[H", end="")

# Parse stream into per-frame byte slices
stream  = raw[hdr_size:]
ptr     = 0
frame_slices = []
for _ in range(total_f):
    if ptr >= len(stream):
        break
    sz = frame_data_size(stream, ptr, w, h)
    frame_slices.append(stream[ptr:ptr + sz])
    ptr += sz

# Sequential decode (frame-order dependency)
prev_f = np.zeros((h, w), dtype=np.uint8)
for i, fdata in enumerate(frame_slices):
    if i >= decode_count:
        break

    decoded  = decode_frame(fdata, prev_f, w, h)
    frame_no = st_i + i
    out_path = os.path.join(args.path, f"{args.suffix}{frame_no:0{pad}d}.bmp")

    Image.fromarray((decoded * 255).astype(np.uint8), mode='L').save(out_path)
    prev_f = decoded.copy()

    p_rat = (i + 1) / total_f
    sys.stdout.write('\033[H')
    print(f"{C_BOLD}BadCodec v0.5.1 Decoder{C_RESET}"
          f"  [{ncpu} cores available]")
    print(_bar_block(p_rat))
    print(f"Frame: {frame_no:04d} / {st_i + total_f - 1} | "
          f"Written: {out_path}")
    print()

print(f"\n{C_BOLD}Decode Complete.{C_RESET}"
      f"  ({decode_count} frames → {args.path}/)")
```

# ============================================================

# CLI

# ============================================================

def main():
p = argparse.ArgumentParser(description=“BadCodec v0.5.1”)
p.add_argument(’-t’, ‘–task’,   choices=[‘e’,‘d’], required=True,
help=“e=encode  d=decode”)
p.add_argument(’-p’, ‘–path’,   required=True,
help=“BMP directory (encode:input / decode:output)”)
p.add_argument(’-n’, ‘–suffix’, default=‘frame_’,
help=“Filename prefix (default: frame_)”)
p.add_argument(’-s’, ‘–start’,  default=‘0001’,
help=“Start frame number (default: 0001)”)
p.add_argument(’-e’, ‘–end’,    default=None,
help=“End frame number (required for encode)”)
p.add_argument(’-o’, ‘–output’, default=‘output.bad’,
help=“Output .bad file (default: output.bad)”)
p.add_argument(’-i’, ‘–input’,  default=None,
help=“Input .bad file (decode only)”)
args = p.parse_args()

```
if args.task == 'e':
    if not args.end:
        print("Error: -e is required for encode."); sys.exit(1)
    do_encode(args)
else:
    do_decode(args)
```

if **name** == ‘**main**’:
main()