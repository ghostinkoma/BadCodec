#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BadCodec v0.5.1
Based on BadCodec 完全実装仕様書 rev.11
  - MD5 -> Fletcher-16 に変更(マイコン向け最低解)

Usage:
  Encode: python badcodec.py -t e -p ./frames -n frame_ -s 0001 -e 6572 -o output.bad
  Decode: python badcodec.py -t d -i output.bad -p ./out  -n frame_ -s 0001
"""

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
C_RESET   = "\033[0m"
C_BOLD    = "\033[1m"
C_GREEN   = "\033[92m"   # S: Skip
C_WHITE   = "\033[97m"   # F: Fill
C_YELLOW  = "\033[33m"   # R: RLE (4-run and 8-run)
C_CYAN    = "\033[36m"   # X: Shift
C_RED     = "\033[91m"   # M: Master
C_MAGENTA = "\033[35m"   # I: Invert
C_BLUE    = "\033[94m"   # D: XOR Diff

TYPE_COLOR = {
    'S': C_GREEN, 'F': C_WHITE, 'R': C_YELLOW,
    'X': C_CYAN,  'M': C_RED,   'I': C_MAGENTA, 'D': C_BLUE,
}

# ============================================================
# Opcodes  (仕様書 第5章 bit tree より)
# ============================================================
# --- Block-level ---
# 0x00-0x1F : BLOCK_INVERT   (000nnnnn)
# 0x20-0x2F : RLE_BLOCK      (0010pppc)  pp=scan_pattern bit3-1, c=start_color
# 0x30-0x37 : FILL_BLOCK     (00110cnn)  c=color, nn=count-1
# 0x38-0x3F : FRAME_CONTROL  (00111xxx)  ※ block context では MASTER_BLOCK=0x3C のみ使用
# 0x40-0x7F : SHIFT_BIT      (01sxxcyy)  s=sign_x, xx=mag_x, c=sign_y, yy=mag_y
# 0x80-0xBF : SKIP_BLOCK     (10nnnnnn)
# 0xC0-0xFF : FOR            (11nnnnnn)

# --- Frame-level opcodes (0x38-0x3F + 0x30) ---
OP_RLE_FRAME        = 0x30   # フレーム全体 RLE 8パターン (rev.12)
OP_DELTA_FRAME      = 0x3D   # XOR差分フレーム + RLE 8パターン (rev.16)
                              # curr XOR prev を RLE_FRAME と同形式で格納
OP_FRAME_DELIMITER  = 0x38
OP_SKIP_FRAME       = 0x39
OP_FRAME_FILL_BLACK = 0x3A
OP_FRAME_FILL_WHITE = 0x3B
OP_BLOCK_STREAM     = 0x3C   # ブロック命令列開始 (rev.12)
OP_MASTER_FRAME     = 0x3E
OP_INVERT_PREV      = 0x3F
OP_EXT_PREFIX       = 0xFF

# Block-level opcodes
OP_MASTER_BLOCK_B   = 0x3C   # ブロックコンテキスト内の MASTER_BLOCK
OP_XOR_BLOCK_B      = 0x3F   # ブロックコンテキスト内の XOR_BLOCK

# RLE_BLOCK_8 opcode table (block context, FRAME_CONTROL free slots)
# (opcode, SCAN_PATHS index, start_color)
# 0x38-0x3B: H scan top-start  0x3D-0x3E: V scan top-start
# 0x3F: XOR_BLOCK (see below)
RLE8_TABLE = [
    (0x38, 0, 0),   # H, TL(left), black-first
    (0x39, 0, 1),   # H, TL(left), white-first
    (0x3A, 1, 0),   # H, TR(right), black-first
    (0x3B, 1, 1),   # H, TR(right), white-first
    (0x3D, 4, 0),   # V, TL(top), black-first
    (0x3E, 4, 1),   # V, TL(top), white-first
]
RLE8_OPCODES = frozenset(op for op, _, _ in RLE8_TABLE)

# XOR_BLOCK: block context only (0x3F)
# Frame context: 0x3F = INVERT_PREV_FRAME (no conflict)
OP_XOR_BLOCK_B = 0x3F

# ============================================================
# Constants
# ============================================================
BLOCK_SIZE   = 8
VERSION      = 514       # protocol version rev.18
APP_VERSION  = "0.6.0"  # FOR min repeat 4→3 (0xC1), max stays 65

# ============================================================
# Fletcher-16  (仕様書 3-3節)
# マイコン向け最低解チェックサム
#   s1 = Σ data[i]      (mod 256)
#   s2 = Σ s1           (mod 256)
#   格納値 = (s2 << 8) | s1  リトルエンディアン2バイト
#
# コード量 : 数十バイト(MD5の1/20)
# RAM      : 2バイト(MD5の1/32)
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
# [2: hdr_size] [2: Fletcher-16] [3: "Bad"] [2: ver] [2: colors]
# [2: w] [2: h] [2: blksize] [2: total_frames]
# ヘッダー全体サイズ: 2 + 2 + 15 = 19バイト固定
# ============================================================
def encode_header(w, h, total_frames):
    body = bytearray()
    body.extend(b'Bad')
    body.extend(struct.pack('<HHHHHH', VERSION, 2, w, h, BLOCK_SIZE, total_frames))
    chk  = fletcher16(bytes(body))
    size = 2 + 2 + len(body)   # hdr_size(2) + Fletcher-16(2) + body(15)
    out  = bytearray()
    out.extend(struct.pack('<H', size))
    out.extend(struct.pack('<H', chk))
    out.extend(body)
    return bytes(out)

def decode_header(data):
    """Returns (w, h, block_size, total_frames, hdr_size) or raises ValueError."""
    if len(data) < 4:
        raise ValueError("File too short for header.")
    hdr_size   = struct.unpack('<H', data[0:2])[0]
    stored_chk = struct.unpack('<H', data[2:4])[0]
    body       = data[4:hdr_size]
    calc_chk   = fletcher16(bytes(body))
    if stored_chk != calc_chk:
        raise ValueError(
            f"Fletcher-16 Mismatch! "
            f"stored=0x{stored_chk:04X} calc=0x{calc_chk:04X} "
            f"File may be corrupted.")
    if body[:3] != b'Bad':
        raise ValueError(f"Invalid magic number: {body[:3]}")
    _ver, _colors, w, h, blk, total_f = struct.unpack('<HHHHHH', body[3:15])
    return w, h, blk, total_f, hdr_size

# ============================================================
# Scan paths for RLE_BLOCK  (仕様書 6-5節)
# Pattern index  = (scan_dir << 2) | start_pos_bits
#   scan_dir   : 0=horizontal, 1=vertical
#   start_pos  : bit1=start_y(0=top,1=bottom), bit0=start_x(0=left,1=right)
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
# RLE_BLOCK_8 bit-pack / unpack  (仕様書 6-6節)
# 6bit × 8 runs → 6 bytes  (3バイトパックを2回繰り返す構造)
# ============================================================
def pack_rle8(runs):
    r = (list(runs) + [0]*8)[:8]
    r = [int(x) & 0x3F for x in r]
    b0 = r[0] | ((r[1] & 0x03) << 6)
    b1 = ((r[1] >> 2) & 0x0F) | ((r[2] & 0x0F) << 4)
    b2 = ((r[2] >> 4) & 0x03) | (r[3] << 2)
    b3 = r[4] | ((r[5] & 0x03) << 6)
    b4 = ((r[5] >> 2) & 0x0F) | ((r[6] & 0x0F) << 4)
    b5 = ((r[6] >> 4) & 0x03) | (r[7] << 2)
    return bytes([b0, b1, b2, b3, b4, b5])

def unpack_rle8(data):
    b0,b1,b2,b3,b4,b5 = data[0],data[1],data[2],data[3],data[4],data[5]
    r0 = b0 & 0x3F
    r1 = ((b0 >> 6) & 0x03) | ((b1 & 0x0F) << 2)
    r2 = ((b1 >> 4) & 0x0F) | ((b2 & 0x03) << 4)
    r3 = (b2 >> 2) & 0x3F
    r4 = b3 & 0x3F
    r5 = ((b3 >> 6) & 0x03) | ((b4 & 0x0F) << 2)
    r6 = ((b4 >> 4) & 0x0F) | ((b5 & 0x03) << 4)
    r7 = (b5 >> 2) & 0x3F
    return [r0, r1, r2, r3, r4, r5, r6, r7]

# ============================================================
# SHIFT_BIT helpers  (仕様書 6-3節)
# ============================================================
def apply_shift(block, sx, sy):
    """
    Shift 8x8 block by (sx, sy).
    Padding: edge pixel value of the pre-shift block.
    +X = right, -X = left, +Y = down, -Y = up
    """
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
    """Try all 8 scan patterns × 2 start colors. Return (opcode, 3bytes) or None."""
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

# ============================================================
# RLE_BLOCK_8 encoder  (仕様書 6-6節)
# _try_rle が失敗(5-8ランが必要)な場合のみ呼ばれる
# ============================================================
def _try_rle8(block):
    """
    RLE8_TABLE の6パターンを試し, ランが 5-8 個に収まる最初の
    パターンを (opcode, bytes6) で返す。全て失敗なら None。
    """
    for opcode, p_idx, start_col in RLE8_TABLE:
        path   = SCAN_PATHS[p_idx]
        pixels = block[path[:, 1], path[:, 0]]
        runs   = []
        curr   = start_col
        count  = 0
        valid  = True
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
        if len(runs) > 8:
            continue
        return (opcode, pack_rle8(runs))
    return None

# ============================================================
# XOR_BLOCK encoder  (仕様書 6-8節)
# curr と prev の XOR マスクを RLE エンコードして返す
# ============================================================
def _xor_block_rle(curr_b, prev_b):
    """
    XOR マスク (64bit) を RLE エンコードして返す。
    形式: (color<<7 | run_length) の列, 1-63 ラン。
    """
    xor_flat = (curr_b.astype(np.uint8) ^ prev_b.astype(np.uint8)).flatten()
    out = bytearray()
    i   = 0
    while i < 64:
        color = int(xor_flat[i])
        count = 1
        while i + count < 64 and int(xor_flat[i + count]) == color and count < 63:
            count += 1
        out.append((color << 7) | count)
        i += count
    return bytes(out)

def encode_block(curr_b, prev_b):
    """
    Encode one 8x8 block. Returns (bytes, type_char).

    全候補を独立に計算し最小バイト数を採用する。
    早期終了による分岐は一切行わない。
    1バイト命令は数学的に常に最小なので、
    1バイト候補が1つでも見つかれば多バイト候補の計算を省略する。

    採用優先順位（同サイズ時）:
      S > X > F > I > R > D > M
      S を最優先とすることで FOR-merge の効率を最大化する。
    """
    # ============================================================
    # Phase-1: 1バイト候補を全て独立に計算する
    # ============================================================
    one_byte = []   # (bytes, type_char)

    # SKIP_BLOCK: prev と完全一致
    if np.array_equal(curr_b, prev_b):
        one_byte.append((bytes([0x80]), 'S'))

    # SHIFT_BIT: prev を ±3 ドット全方向シフトして一致するか試算
    # SKIP がある場合も含め常に独立して試算する（but 1Bなので意味は同じ）
    # ただし SKIP が見つかれば SHIFT(1B)でも結果が変わらないので省略可
    if not any(t == 'S' for _, t in one_byte):
        shift_found = False
        for sx in range(-3, 4):
            if shift_found: break
            for sy in range(-3, 4):
                if sx == 0 and sy == 0:
                    continue
                if np.array_equal(apply_shift(prev_b, sx, sy), curr_b):
                    sign_x = 1 if sx < 0 else 0
                    mag_x  = abs(sx)
                    sign_y = 1 if sy < 0 else 0
                    mag_y  = abs(sy)
                    op = 0x40 | (sign_x << 5) | (mag_x << 3) \
                              | (sign_y << 2) | mag_y
                    one_byte.append((bytes([op]), 'X'))
                    shift_found = True
                    break

    # FILL_BLOCK: 全黒 または 全白
    if np.all(curr_b == 0):
        one_byte.append((bytes([0x30]), 'F'))
    elif np.all(curr_b == 1):
        one_byte.append((bytes([0x34]), 'F'))

    # BLOCK_INVERT: prev の全ビット反転と一致
    if np.array_equal(curr_b, 1 - prev_b):
        one_byte.append((bytes([0x00]), 'I'))

    # ============================================================
    # 1バイト命令が見つかれば即返す
    # 優先順: S > F > I > X
    #
    # S を最優先: SKIP は最も長いランを形成しやすく FOR merge に最大寄与
    # F を次位:   FILL も連続しやすく FOR merge に寄与
    #             X より先にすることで FX 同時成立時に F が勝ち
    #             FOR merge の機会を最大化する
    # I を次位:   INVERT も連続しうる
    # X を最後:   SHIFT は移動量がブロック毎に異なるため
    #             ほぼ連続しない → FOR の恩恵を受けにくい
    # ============================================================
    if one_byte:
        for preferred_type in ('S', 'F', 'I', 'X'):
            for cand in one_byte:
                if cand[1] == preferred_type:
                    return cand
        return one_byte[0]

    # ============================================================
    # Phase-2: 多バイト候補を全て独立に計算し最小を選ぶ
    # 1バイト命令が存在しない場合のみ到達する
    # ============================================================
    candidates = []

    # RLE_BLOCK_4 (4B): 4ラン以内に収まる場合
    rle4 = _try_rle(curr_b)
    if rle4 is not None:
        candidates.append((bytes([rle4[0]]) + rle4[1], 'R'))   # 4B

    # XOR_BLOCK (2+N B): prev XOR curr のRLE
    # MASTER(9B)より小さい場合のみ候補に加える
    xor_rle   = _xor_block_rle(curr_b, prev_b)
    xor_total = 2 + len(xor_rle)
    if xor_total < 9:
        candidates.append(
            (bytes([OP_XOR_BLOCK_B, len(xor_rle)]) + xor_rle, 'D'))

    # RLE_BLOCK_8 (7B): 5-8ランが必要な場合
    # RLE4(4B) が成功していれば 7B は勝てないので試算不要
    if rle4 is None:
        rle8 = _try_rle8(curr_b)
        if rle8 is not None:
            candidates.append((bytes([rle8[0]]) + rle8[1], 'R'))  # 7B

    # MASTER_BLOCK (9B): 無条件フォールバック
    candidates.append((_encode_master_block(curr_b), 'M'))

    # 最小バイト数を採用（同サイズなら先に追加された候補が勝つ）
    return min(candidates, key=lambda c: len(c[0]))

# ============================================================
# Multi-block merge  (SKIP / INVERT / FILL 連続をまとめる)
# ============================================================
def _best_for_skip(N, base_op, max_inner, max_for=65):
    """
    N ブロックのランを FOR(n)+CMD(i) + 余り で最小バイト数にエンコードする。

    引数:
      N        : エンコードすべきブロック数
      base_op  : 命令のベースオペコード (SKIP=0x80, FILL_BLACK=0x30 等)
      max_inner: 命令1回あたりの最大ブロック数 (SKIP=64, INVERT=32, FILL=4)
      max_for  : FOR の最大繰り返し数 (仕様上 65)

    返値: bytearray

    アルゴリズム:
      余りコストを naive_bytes ではなく再帰的に最適計算することで
      FOR を複数段重ねた最適解を確実に発見する。
        例: N=68, max_inner=1
          旧(naive): FOR(65)+CMD + 3literals = 5B
          新(再帰):  FOR(34)+CMD + FOR(34)+CMD = 4B  ← 余り34もFOR化できる

      FOR禁止ルール:
        FOR(0xC0) = repeat 2 → 禁止 (直書き2Bと等コスト・無意味)
        FOR(0xC1) = repeat 3 → 有効 (2B < 直書き3B で 1B 節約) ★
        よって禁止は repeat=2 のみ。最小有効値は repeat=3 (k >= 3)
    """
    if N <= 0:
        return bytearray()

    import math

    # ---- 再帰的最小コスト計算 --------------------------------
    def min_cost(n):
        """n ブロックを表現する最小バイト数を再帰的に計算する"""
        if n <= 0:
            return 0
        best = math.ceil(n / max_inner)   # 直書きコスト（FOR なし）
        for i in range(1, max_inner + 1):
            for k in (n // i, n // i + 1):
                # repeat=2(0xC0)のみ禁止: 直書き2Bと同コストで無意味
                # repeat=3(0xC1)は有効: FOR+CMD=2B < 直書き3B
                if k < 3 or k > max_for:
                    continue
                product = k * i
                if product > n:
                    continue
                cost = 2 + min_cost(n - product)
                if cost < best:
                    best = cost
        return best

    # ---- 最適な (k, i) を探索 --------------------------------
    best_cost  = math.ceil(N / max_inner)
    best_tuple = None

    for i in range(1, max_inner + 1):
        # k を 3〜min(N//i, max_for) の全値で試算する
        # repeat=3 (0xC1) が最小有効値
        k_max = min(N // i, max_for)
        for k in range(3, k_max + 1):
            product   = k * i
            if product > N:
                continue
            remainder = N - product
            cost = 2 + min_cost(remainder)   # 余りも再帰的に最適計算
            if cost < best_cost:
                best_cost  = cost
                best_tuple = (k, i, remainder)

    # ---- エンコード出力生成 ----------------------------------
    out = bytearray()
    if best_tuple is None:
        # FOR なし直書き
        rem = N
        while rem > 0:
            take = min(rem, max_inner)
            out.append(base_op | (take - 1))
            rem -= take
    else:
        k, i, remainder = best_tuple
        for_byte = 0xC0 | (k - 2)
        inner_op = base_op | (i - 1)
        out.extend([for_byte, inner_op])
        # 余りを再帰的にエンコード
        out.extend(_best_for_skip(remainder, base_op, max_inner, max_for))

    return out

# ============================================================
# Multi-block merge  (SKIP / INVERT / FILL 連続をまとめる)
# FOR(n)+CMD(i) 最適化を内包
# ============================================================
def _merge_multiblock(raw_cmds, raw_types):
    result_c, result_t = [], []
    i = 0
    while i < len(raw_cmds):
        t = raw_types[i]
        c = raw_cmds[i]

        # ---- SKIP_BLOCK ランを集約 -------------------------
        if t == 'S' and c == bytes([0x80]):
            j = i
            while j < len(raw_cmds) and raw_types[j] == 'S' \
                    and raw_cmds[j] == bytes([0x80]):
                j += 1
            N   = j - i
            buf = _best_for_skip(N, 0x80, 64)
            # FOR命令が含まれる場合は1トークンとして扱う
            result_c.append(bytes(buf))
            result_t.append('S')
            i = j

        # ---- BLOCK_INVERT ランを集約 -----------------------
        elif t == 'I' and c == bytes([0x00]):
            j = i
            while j < len(raw_cmds) and raw_types[j] == 'I' \
                    and raw_cmds[j] == bytes([0x00]):
                j += 1
            N   = j - i
            buf = _best_for_skip(N, 0x00, 32)
            result_c.append(bytes(buf))
            result_t.append('I')
            i = j

        # ---- FILL_BLOCK ランを集約 -------------------------
        elif t == 'F' and len(c) == 1:
            base = c[0] & 0xFC   # 色ビット保持、カウントビットを落とす
            j    = i
            while j < len(raw_cmds) and raw_types[j] == 'F' \
                    and len(raw_cmds[j]) == 1 \
                    and (raw_cmds[j][0] & 0xFC) == base:
                j += 1
            N   = j - i
            buf = _best_for_skip(N, base, 4)
            result_c.append(bytes(buf))
            result_t.append('F')
            i = j

        else:
            result_c.append(c)
            result_t.append(t)
            i += 1

    return result_c, result_t

# ============================================================
# FOR optimizer  (仕様書 9-2節)
# S/F/I ランは _merge_multiblock が既に FOR 最適化済み。
# X/R/D/M 等の単一バイト命令ランも _best_for_skip で最適化する。
# max_inner=1 (これらの命令は1回で1ブロックのみ消費)
# FOR 最小 repeat=3 (0xC1): FOR+CMD=2B < 直書き3B
# FOR 禁止 repeat=2 (0xC0): FOR+CMD=2B = 直書き2B (等コスト・無意味)
# ============================================================
def optimize_for(cmds, types):
    """
    Merge consecutive identical single-byte commands using FOR.
    Uses _best_for_skip (N=n*i+j exhaustive search) for all runs.
    Multi-byte tokens from _merge_multiblock pass through as-is.
    """
    out_b = bytearray()
    out_t = []
    i     = 0
    while i < len(cmds):
        cmd = cmds[i]
        t   = types[i]
        # マルチバイトトークン (S/F/I の FOR 最適化済みバイト列) はそのまま通過
        if len(cmd) != 1:
            out_b.extend(cmd)
            out_t.append(t)
            i += 1
            continue
        # 連続する同一単一バイト命令のランを数える
        j = i + 1
        while j < len(cmds) and cmds[j] == cmd and len(cmds[j]) == 1:
            j += 1
        N   = j - i
        # _best_for_skip で最適エンコード (max_inner=1)
        # base_op = cmd[0], max_inner = 1 (1命令=1ブロック)
        buf = _best_for_skip(N, cmd[0], 1)
        out_b.extend(buf)
        out_t.append(t)
        i = j
    return bytes(out_b), out_t

# ============================================================
# RLE_FRAME encoder / decoder  (仕様書 第7章)
#
# オペコード 0x30-0x37 = 8走査パターン
#   bit2 : 走査方向  (0=横優先, 1=縦優先)
#   bit1 : 開始Y     (0=y=0,    1=y=height_max)
#   bit0 : 開始X     (0=x=0,    1=x=width_max)
#
# エンコード時に全8パターンを試算し最小バイト数を採用する。
# ============================================================

def _rle_scan_pixels(frame, scan_dir, start_y, start_x):
    """
    フレームを指定走査パターンで1次元化して返す (numpy 版)。
    flip + 転置の組み合わせで8パターンを O(1) で生成。
    """
    f = frame
    if start_x: f = f[:, ::-1]   # 水平反転
    if start_y: f = f[::-1, :]   # 垂直反転
    if scan_dir: f = f.T          # 縦優先は転置 (h,w) -> (w,h)
    return f.flatten()

def _rle_encode_pixels(pixels):
    """1次元ピクセル列を RLE バイト列に変換。"""
    out = bytearray()
    i = 0
    n = len(pixels)
    while i < n:
        color = int(pixels[i])
        count = 1
        while i + count < n and int(pixels[i + count]) == color and count < 127:
            count += 1
        out.append((color << 7) | count)
        i += count
    return bytes(out)

def encode_rle_frame_best(frame):
    """
    8パターン全試算し最小バイト数の (opcode + rle_data) を返す。
    0x30 (横/TL) は旧 RLE_FRAME と同一なので後方互換を保つ。
    """
    best_bytes = None
    best_op    = 0x30
    for scan_dir in range(2):
        for start_y in range(2):
            for start_x in range(2):
                op  = 0x30 | (scan_dir << 2) | (start_y << 1) | start_x
                pix = _rle_scan_pixels(frame, scan_dir, start_y, start_x)
                rle = _rle_encode_pixels(pix)
                if best_bytes is None or len(rle) < len(best_bytes):
                    best_bytes = rle
                    best_op    = op
    return bytes([best_op]) + best_bytes

# 旧名互換エイリアス (decode_frame から呼ばれる)
def encode_rle_frame(frame_2d):
    """シンプルな横/TL固定 (0x30)。テスト・後方互換用。"""
    pix = _rle_scan_pixels(frame_2d, 0, 0, 0)
    return _rle_encode_pixels(pix)

# ============================================================
# Frame-level decode  (仕様書 第7章)
# ============================================================
def decode_frame(data, prev_f, w, h):
    bx     = w // BLOCK_SIZE
    by_cnt = h // BLOCK_SIZE
    n_blk  = bx * by_cnt
    op     = data[0]

    # Frame-level FOR: 0xC0-0xFF
    # merge_frame_for が生成する FOR(n)+SINGLE_BYTE_OP の形式
    # repeat は呼び出し元(do_decode)が管理するため
    # ここでは inner_op を1回分デコードして返すだけで正しい
    if (op & 0xC0) == 0xC0:
        if len(data) < 2:
            raise ValueError(f"Frame-level FOR: insufficient data")
        inner_op = data[1]
        return decode_frame(bytes([inner_op]), prev_f, w, h)

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
    # RLE_FRAME: 0x30-0x37 (8走査パターン)
    if 0x30 <= op <= 0x37:
        scan_dir = (op >> 2) & 1
        start_y  = (op >> 1) & 1
        start_x  =  op       & 1
        return _decode_rle_frame(data[1:], w, h, scan_dir, start_y, start_x)
    # DELTA_FRAME: 0x3D  XOR差分 + RLE (rev.16)
    # data[0]=0x3D, data[1]=走査パターン(0x30-0x37と同形式), data[2:]=RLEデータ
    if op == OP_DELTA_FRAME:
        pat      = data[1]
        scan_dir = (pat >> 2) & 1
        start_y  = (pat >> 1) & 1
        start_x  =  pat       & 1
        diff_f   = _decode_rle_frame(data[2:], w, h, scan_dir, start_y, start_x)
        return (prev_f ^ diff_f).astype(np.uint8)
    if op == OP_BLOCK_STREAM:
        return _decode_block_stream(data[1:], prev_f, w, h, bx, n_blk)

    raise ValueError(f"Unknown frame opcode: 0x{op:02X}")

def _decode_rle_frame(rle, w, h, scan_dir=0, start_y=0, start_x=0):
    """
    RLE バイト列を走査パターンに従ってデコードし (h, w) フレームを返す。

    エンコード時の変換:
      1. start_x が 1 なら水平反転
      2. start_y が 1 なら垂直反転
      3. scan_dir が 1 なら転置 (縦優先)
      4. flatten して RLE

    デコードは逆順で復元する。
    """
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

    # 逆変換: scan_dir=1 は転置状態なので (w,h) にリシェイプ後転置
    if scan_dir:
        frame = arr.reshape(w, h).T   # (w,h) -> (h,w)
    else:
        frame = arr.reshape(h, w)

    # flip を元に戻す (エンコードと逆順)
    if start_y: frame = frame[::-1, :]
    if start_x: frame = frame[:, ::-1]

    return frame.astype(np.uint8)

def _decode_block_stream(stream, prev_f, w, h, bx, n_blk):
    curr_f = np.zeros((h, w), dtype=np.uint8)
    ptr    = 0
    b_idx  = 0

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

        # RLE_BLOCK_4 (4-run)
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
            blk = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=np.uint8)
            for idx, v in enumerate(pix[:64]):
                blk[path[idx, 1], path[idx, 0]] = v   # path[i]=(x,y) → blk[y,x]
            curr_f[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE] = blk
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

        # RLE_BLOCK_8 (8-run, 6 data bytes)
        if cmd in RLE8_OPCODES:
            entry = next(e for e in RLE8_TABLE if e[0] == cmd)
            _, p_idx, start_col = entry
            runs  = unpack_rle8(stream[p:p+6])
            path  = SCAN_PATHS[p_idx]
            pix   = []
            col   = start_col
            for r in runs:
                pix.extend([col] * r)
                col = 1 - col
            blk = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=np.uint8)
            for idx, v in enumerate(pix[:64]):
                blk[path[idx, 1], path[idx, 0]] = v   # path[i]=(x,y) → blk[y,x]
            curr_f[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE] = blk
            return b_i + 1, p + 6

        # MASTER_BLOCK
        if cmd == OP_MASTER_BLOCK_B:
            bits = np.unpackbits(np.frombuffer(stream[p:p+8], dtype=np.uint8))
            curr_f[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE] = \
                bits.reshape(BLOCK_SIZE, BLOCK_SIZE)
            return b_i + 1, p + 8

        # XOR_BLOCK (variable length: length_byte + RLE_data)
        if cmd == OP_XOR_BLOCK_B:
            xor_len = stream[p]; p += 1
            xor_pix = []
            j = p
            while len(xor_pix) < 64 and j < p + xor_len:
                b    = stream[j]; j += 1
                col  = (b >> 7) & 1
                run  = b & 0x7F
                xor_pix.extend([col] * min(run, 64 - len(xor_pix)))
            xor_mask = np.array(xor_pix[:64], dtype=np.uint8).reshape(BLOCK_SIZE, BLOCK_SIZE)
            prev_blk = prev_f[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE]
            curr_f[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE] = (prev_blk ^ xor_mask).astype(np.uint8)
            return b_i + 1, p + xor_len

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

        # Unknown -- skip 1 block
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

# ============================================================
# Frame size calculator  (stream parser に必要)
# ============================================================
def frame_data_size(data, offset, w, h):
    """Return byte size of one frame starting at data[offset]."""
    op = data[offset]
    # Frame-level FOR: 0xC0-0xFF
    # merge_frame_for が生成する FOR(n)+SINGLE_BYTE_OP = 2バイト固定
    if (op & 0xC0) == 0xC0:
        return 2
    # Fixed 1-byte frames
    if op in (OP_SKIP_FRAME, OP_FRAME_FILL_BLACK,
              OP_FRAME_FILL_WHITE, OP_INVERT_PREV):
        return 1
    # MASTER_FRAME: 1 + w*h/8
    if op == OP_MASTER_FRAME:
        return 1 + (w * h) // 8
    # RLE_FRAME: 0x30-0x37 (8走査パターン共通)
    if 0x30 <= op <= 0x37:
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
    # DELTA_FRAME: 0x3D  opcode(1) + pattern(1) + RLEデータ(可変)
    if op == OP_DELTA_FRAME:
        total  = w * h
        count  = 0
        i      = offset + 2   # opcode + pattern
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
            # XOR_BLOCK: variable length (special case)
            if cmd == OP_XOR_BLOCK_B:
                if ptr < len(data):
                    xor_len = data[ptr]; ptr += 1
                    ptr += xor_len
                b_idx += 1
                continue
            b_idx += _block_advance(cmd)
            ptr   += _block_extra_bytes(cmd)
        return ptr - offset
    return 1

def _block_advance(cmd):
    """How many blocks does this command consume?"""
    if cmd & 0x80:              return (cmd & 0x3F) + 1   # SKIP_BLOCK
    if cmd <= 0x1F:             return (cmd & 0x1F) + 1   # BLOCK_INVERT
    if 0x20 <= cmd <= 0x2F:    return 1                    # RLE_BLOCK_4
    if 0x30 <= cmd <= 0x37:    return (cmd & 0x03) + 1    # FILL_BLOCK
    if cmd in RLE8_OPCODES:    return 1                    # RLE_BLOCK_8
    if cmd == OP_MASTER_BLOCK_B: return 1                  # MASTER_BLOCK
    # OP_XOR_BLOCK_B (0x3F) is variable-length; handled separately
    if 0x40 <= cmd <= 0x7F:    return 1                    # SHIFT_BIT
    return 1

def _block_extra_bytes(cmd):
    """How many fixed extra data bytes follow this command? (variable-length handled separately)"""
    if 0x20 <= cmd <= 0x2F:    return 3   # RLE_BLOCK_4: 3 bytes
    if cmd in RLE8_OPCODES:    return 6   # RLE_BLOCK_8: 6 bytes
    if cmd == OP_MASTER_BLOCK_B: return 8  # MASTER_BLOCK: 8 bytes
    # OP_XOR_BLOCK_B (0x3F): variable, handled by frame_data_size directly
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
        err = "verify mismatch"
    except Exception as e:
        err = str(e)

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

# ============================================================
# Frame encoder worker  (multiprocessing)
# ============================================================
def encode_frame_worker(args):
    frame_idx, curr_path, prev_path, w, h = args

    curr_f = np.array(Image.open(curr_path).convert('1'), dtype=np.uint8)
    if prev_path and os.path.exists(prev_path):
        prev_f = np.array(Image.open(prev_path).convert('1'), dtype=np.uint8)
    else:
        prev_f = np.zeros((h, w), dtype=np.uint8)

    bx     = w // BLOCK_SIZE
    n_blk  = bx * (h // BLOCK_SIZE)
    raw_sz = (w * h) // 8          # MASTER_FRAME のデータ部固定サイズ

    # ==========================================================
    # 全候補を収集して最小バイト数を選ぶ
    # candidates: list of (encoded_bytes, type_list)
    # ==========================================================
    candidates = []

    # ----------------------------------------------------------
    # Phase-1: フレーム単一命令 (1B確定)
    # これらが1つでも成立すれば他の全候補に勝つ (1B < 2B)
    # 早期終了して計算コストを節約する
    # ----------------------------------------------------------
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

    # ----------------------------------------------------------
    # Phase-2: 固定長候補を先に計算
    # MASTER_FRAME のサイズは固定 (raw_sz + 1) なので先に確定できる
    # これを上限として BLOCK_STREAM の採否を早期判定する
    # ----------------------------------------------------------

    # 候補C: MASTER_FRAME (固定長 raw_sz+1 B)
    cand_c = bytes([OP_MASTER_FRAME]) + bytes(np.packbits(curr_f.flatten()))
    candidates.append((cand_c, ['M'] * n_blk))
    master_sz = len(cand_c)   # = raw_sz + 1

    # ----------------------------------------------------------
    # Phase-3: RLE系候補 (curr_f / diff_f の両方を試算)
    # BLOCK_STREAMより計算コストが低いので先に実施する
    # ----------------------------------------------------------

    # 候補B: RLE_FRAME (curr_f に対して8走査パターン全試算)
    cand_b = encode_rle_frame_best(curr_f)
    candidates.append((cand_b, ['R'] * n_blk))

    # 候補E: DELTA_FRAME (curr XOR prev を RLE_FRAME と同形式で格納)
    # フォーマット: OP_DELTA_FRAME(1B) + pattern(1B) + RLEデータ
    diff_f    = (curr_f ^ prev_f).astype(np.uint8)
    best_diff = encode_rle_frame_best(diff_f)   # (op + rle_data)
    # best_diff[0] は 0x30-0x37 なので pattern byte として再利用
    cand_e = bytes([OP_DELTA_FRAME, best_diff[0]]) + best_diff[1:]
    candidates.append((cand_e, ['D'] * n_blk))

    # ----------------------------------------------------------
    # Phase-4: BLOCK_STREAM
    # 全ブロックエンコードが必要で最も計算コストが高い
    # RLE/MASTERが既に十分小さい場合でも常に試算して正確に比較する
    # (エンコード時間より圧縮率を優先する設計)
    # ----------------------------------------------------------
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

    # 候補A: BLOCK_STREAM
    cand_a = bytes([OP_BLOCK_STREAM]) + opt_bytes
    candidates.append((cand_a, opt_types))

    # ----------------------------------------------------------
    # 全候補から最小バイト数を採用
    # 同サイズの場合: A(BLOCK) > B(RLE) > C(MASTER) > E(DELTA)
    # の逆順でリストに入っているので先勝ちではなく len() で比較
    # ----------------------------------------------------------
    best_data, best_types = min(candidates, key=lambda x: len(x[0]))

    return _self_verify(frame_idx, best_data, best_types, curr_f, prev_f, w, h)

# ============================================================
# UI helpers
# ============================================================
def _bar_hash(curr, total, width=40):
    r    = curr / total if total > 0 else 0
    done = int(width * r)
    return f"[{'#'*done}{'.'*(width-done)}] {int(r*100):3d}% ({curr}/{total})"

def _bar_block(ratio, width=20):
    # ratio > 1.0 (非圧縮より大きいフレーム) でバーがはみ出ないようにクランプ
    done = min(int(width * ratio), width)
    pct  = ratio * 100
    # 100%超えは色付きで警告表示
    if pct > 100.0:
        return f"{C_RED}{'█'*done}{'░'*(width-done)}{C_RESET} {pct:5.1f}%"
    return f"{'█'*done}{'░'*(width-done)} {pct:4.1f}%"

def _block_map(types, bx, max_rows=8, max_cols=30):
    rows = min(max_rows, len(types) // bx if bx > 0 else 0)
    cols = min(max_cols, bx)
    lines = []
    for r in range(rows):
        row = ""
        for c in range(cols):
            idx = r * bx + c
            t   = types[idx] if idx < len(types) else '?'
            row += f"{TYPE_COLOR.get(t, C_RESET)}{t}{C_RESET} "
        lines.append(row)
    return lines

# ============================================================
# Frame-level FOR merge pass  (仕様書 9-2節・13章)
#
# 判定ルール：
#   フレーム命令を素直に書いて良いのは
#     「直前がフレーム命令でない」かつ「直前と同じフレーム命令でない」
#   それ以外は FOR への包含を検討する
#
# FOR 禁止：FOR 0(2回)・FOR 1(3回) → N≦3 は個別に書く
# FOR 上限：65回 → 65を超える場合は次のFORブロックに分割
# ============================================================
SINGLE_BYTE_FRAME_OPS = {
    OP_SKIP_FRAME,
    OP_FRAME_FILL_BLACK,
    OP_FRAME_FILL_WHITE,
    OP_INVERT_PREV,
}

def merge_frame_for(ordered_frames):
    """
    ordered_frames: list of (data_bytes, blk_types)
    全フレームを順番に受け取り、連続するフレーム命令を FOR でまとめる。

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

            # 連続する同じ命令を数える (FOR 上限 65 まで)
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
                # FOR禁止ルール: repeat=2 (0xC0) のみ禁止
                #   repeat=2: FOR+CMD=2B = 直書き2B → 等コスト・無意味
                #   repeat=3: FOR+CMD=2B < 直書き3B → 1B 節約・有効 ★
                #   よって take<3 のみ直書き
                if take < 3:
                    for k in range(take):
                        out.extend([op])
                        display.append((bytes([op]), types))
                else:
                    # FOR(0xC1以上) + op で表現
                    # 最小有効値: take=3 → 0xC0|(3-2)=0xC1
                    for_byte = 0xC0 | (take - 2)
                    out.extend([for_byte, op])
                    display.append((bytes([for_byte, op]), types))
                rem -= take
            i = j

        else:
            # BLOCK_STREAM / RLE_FRAME / DELTA_FRAME / MASTER_FRAME はそのまま
            # バグ5修正: types がブロックマップ用でない場合（RLE/DELTA/MASTER）は
            # フレーム全体を単一タイプとして扱う（_block_map での表示は一様になる）
            out.extend(data)
            display.append((data, types))
            i += 1

    return bytes(out), display


# ============================================================
# Encoder
# ============================================================
def do_encode(args):
    st_i    = int(args.start)
    ed_i    = int(args.end)
    pad     = len(args.start)
    total_f = ed_i - st_i + 1

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
                avg    = sum(win_90) / len(win_90)
                curr_c = len(ordered_frames)
                p_rat  = curr_c / total_f

                sys.stdout.write('\033[H')
                # バグ1修正: APP_VERSION を使用
                print(f"{C_BOLD}BadCodec v{APP_VERSION} Encoder{C_RESET}"
                      f"  [{ncpu} cores]  Phase 1/2: Encoding")
                print(_bar_block(p_rat))
                # バグ2修正: 実際のフレーム番号 (write_ptr はバッファ内インデックス)
                frame_no_display = write_ptr
                print(f"Frame: {frame_no_display:04d} | "
                      f"Size: {len(bd):6d}B | "
                      f"Avg(90f): {avg:5.2f}% raw")

                # バグ3修正: _block_map を1回だけ呼ぶ
                block_lines = _block_map(bt, bx)
                for line in block_lines:
                    print(line)
                shown = len(block_lines)
                for _ in range(min(8, h // BLOCK_SIZE) - shown):
                    print()

                write_ptr += 1

    # -------------------------------------------------------
    # Phase 2: フレームFOR最適化マージ
    #   連続する同一フレーム命令を FOR に包含する。
    #   この処理は全フレームが確定してからでないと正確に行えない。
    # -------------------------------------------------------
    sys.stdout.write('\033[H')
    print(f"{C_BOLD}BadCodec v{APP_VERSION} Encoder{C_RESET}"
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
    before_merge = sum(len(d) for d,_ in ordered_frames)
    print(f"{C_BOLD}Done.{C_RESET} Saved to {args.output} "
          f"({total_bytes:,} bytes, "
          f"{total_bytes / raw_total * 100:.2f}% of raw  "
          f"{raw_total / total_bytes:.3f}x)")
    print(f"  Frame FOR merge: {before_merge:,}B → {len(merged_stream):,}B"
          f"  (saved {before_merge - len(merged_stream):,}B)")

    if errors:
        print(f"\n{C_RED}Warnings ({len(errors)}):{C_RESET}")
        for e in errors[:10]:
            print(f"  {e}")

# ============================================================
# Decoder
# ============================================================
def do_decode(args):
    if not args.input:
        print("Error: -i is required for decode."); sys.exit(1)
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found."); sys.exit(1)

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
    print(f"{C_BOLD}BadCodec v{APP_VERSION} Decoder{C_RESET}")
    print(f"Input : {args.input}")
    print(f"Output: {os.path.join(args.path, args.suffix)}%04d.bmp")
    print(f"Frames: {total_f}  Size: {w}x{h}")
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
        print(f"{C_BOLD}BadCodec v{APP_VERSION} Decoder{C_RESET}"
              f"  [{ncpu} cores available]")
        print(_bar_block(p_rat))
        print(f"Frame: {frame_no:04d} / {st_i + total_f - 1:04d} | "
              f"Written: {out_path}")
        print()

    print(f"\n{C_BOLD}Decode Complete.{C_RESET}"
          f"  ({decode_count} frames → {args.path}/)")

# ============================================================
# C Header exporter  (-t c)
# ============================================================
def _c_ident(name):
    """ファイル名から有効な C 識別子を生成する"""
    import re
    base = os.path.splitext(os.path.basename(name))[0]
    ident = re.sub(r'[^0-9A-Za-z_]', '_', base)
    if ident and ident[0].isdigit():
        ident = '_' + ident
    return ident or 'bad_data'

def do_c_header(args):
    """
    .bad ファイルを C 言語ヘッダファイルに変換する。

    出力形式:
      #ifndef UPPER_NAME_H
      #define UPPER_NAME_H

      #include <stdint.h>
      #include <avr/pgmspace.h>  // AVR のみ

      // メタデータ定数 (ヘッダから取得)
      #define ARRAY_NAME_WIDTH   128
      #define ARRAY_NAME_HEIGHT   64
      #define ARRAY_NAME_FRAMES 6572
      #define ARRAY_NAME_SIZE   972231

      #ifdef __AVR__
      const uint8_t ARRAY_NAME[] PROGMEM = {
      #else
      const uint8_t ARRAY_NAME[] = {
      #endif
          0x13, 0x00, ...
      };

      #endif // UPPER_NAME_H
    """
    if not args.input:
        print("Error: -i is required for -t c."); sys.exit(1)
    if not args.H:
        print("Error: -H is required for -t c."); sys.exit(1)
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found."); sys.exit(1)

    # 配列名の決定: -n が指定されていれば使用、なければファイル名から生成
    # -t c では -n を「配列名」として流用する
    with open(args.input, 'rb') as f:
        raw = f.read()

    # ヘッダ解析してメタデータを取得
    try:
        w, h, blk, total_f, hdr_size = decode_header(raw)
    except ValueError as e:
        print(f"Header error: {e}"); sys.exit(1)

    # 配列名・ヘッダガード名の決定
    array_name  = _c_ident(args.H)
    guard_name  = array_name.upper() + '_H'
    file_size   = len(raw)

    # 1行あたりのバイト数
    BYTES_PER_LINE = 16

    lines = []
    lines.append(f"#ifndef {guard_name}")
    lines.append(f"#define {guard_name}")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append("/* BadCodec v{ver} / Protocol {proto} */".format(
        ver=APP_VERSION, proto=VERSION))
    lines.append(f"#define {array_name.upper()}_WIDTH   {w}U")
    lines.append(f"#define {array_name.upper()}_HEIGHT  {h}U")
    lines.append(f"#define {array_name.upper()}_FRAMES  {total_f}U")
    lines.append(f"#define {array_name.upper()}_SIZE    {file_size}UL")
    lines.append("")
    lines.append("#ifdef __AVR__")
    lines.append("#include <avr/pgmspace.h>")
    lines.append(f"const uint8_t {array_name}[] PROGMEM = {{")
    lines.append("#else")
    lines.append(f"const uint8_t {array_name}[] = {{")
    lines.append("#endif")

    # バイト列を16バイト/行でフォーマット
    for offset in range(0, file_size, BYTES_PER_LINE):
        chunk   = raw[offset:offset + BYTES_PER_LINE]
        hex_str = ', '.join(f'0x{b:02X}' for b in chunk)
        if offset + BYTES_PER_LINE < file_size:
            lines.append(f"    {hex_str},")
        else:
            lines.append(f"    {hex_str}")

    lines.append("};")
    lines.append("")
    lines.append(f"#endif /* {guard_name} */")
    lines.append("")

    output_text = '\n'.join(lines)

    with open(args.H, 'w', encoding='ascii') as f:
        f.write(output_text)

    print(f"{C_BOLD}BadCodec v{APP_VERSION} C Header Export{C_RESET}")
    print(f"  Input  : {args.input}  ({file_size:,} bytes)")
    print(f"  Output : {args.H}")
    print(f"  Array  : {array_name}[]  ({file_size:,} bytes)")
    print(f"  Image  : {w}x{h}  {total_f} frames")
    print(f"  Guard  : {guard_name}")
    print(f"  PROGMEM: enabled (#ifdef __AVR__)")
    print(f"\n{C_BOLD}Done.{C_RESET}")

# ============================================================
# CLI
# ============================================================
def main():
    p = argparse.ArgumentParser(
        description=f"BadCodec v{APP_VERSION} (Protocol {VERSION})",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Encode:   Codec.py -t e -p ./frames -n frame_ -s 0001 -e 6572 -o out.bad
  Decode:   Codec.py -t d -i out.bad  -p ./out   -n frame_ -s 0001
  C Header: Codec.py -t c -i out.bad  -H video.h
""")
    p.add_argument('-t', '--task',   choices=['e','d','c'], required=True,
                   help="e=encode  d=decode  c=C header export")
    p.add_argument('-p', '--path',   default=None,
                   help="BMP directory (encode:input / decode:output)")
    p.add_argument('-n', '--suffix', default='frame_',
                   help="Filename prefix (default: frame_)")
    p.add_argument('-s', '--start',  default='0001',
                   help="Start frame number (default: 0001)")
    p.add_argument('-e', '--end',    default=None,
                   help="End frame number (required for encode)")
    p.add_argument('-o', '--output', default='output.bad',
                   help="Output .bad file (default: output.bad)")
    p.add_argument('-i', '--input',  default=None,
                   help="Input .bad file (decode / C header)")
    p.add_argument('-H', '--H',      default=None, dest='H',
                   help="Output C header filename (e.g. video.h)  [-t c]")
    args = p.parse_args()

    if args.task == 'e':
        if not args.end:
            print("Error: -e is required for encode."); sys.exit(1)
        if not args.path:
            print("Error: -p is required for encode."); sys.exit(1)
        do_encode(args)
    elif args.task == 'd':
        if not args.path:
            print("Error: -p is required for decode."); sys.exit(1)
        do_decode(args)
    else:  # 'c'
        do_c_header(args)

if __name__ == '__main__':
    main()
