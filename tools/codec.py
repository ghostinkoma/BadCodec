#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import sys
import argparse
from pathlib import Path
import time
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import Counter
import multiprocessing as mp
from functools import partial

# 定数
BLOCK_SIZE = 8
BLOCK_BYTES = 8

CMD_START_FRAME = 0b00000000
CMD_MASTER_BLOCK = 0b00000001
CMD_RLE_BLOCK_BLACK = 0b00000010
CMD_RLE_BLOCK_WHITE = 0b00000011
CMD_DELTA_FRAME_BIT_RAW = 0b00000100  # raw XOR
CMD_MASTER_FRAME = 0b00000101
CMD_DELTA_FRAME_BIT_RLE = 0b00000110  # RLE XOR
CMD_BLOCK_FILL_BLACK = 0b00100000
CMD_BLOCK_FILL_WHITE = 0b00100001
CMD_BLOCK_INVERT = 0b00100010
CMD_SKIP_BLOCK = 0b10000000
CMD_FOR = 0b10100000
CMD_NOP = 0b11100000

CMD_NAMES = {
    CMD_START_FRAME: "START_FRAME",
    CMD_MASTER_BLOCK: "MASTER_BLOCK",
    CMD_RLE_BLOCK_BLACK: "RLE_BLOCK_BLACK",
    CMD_RLE_BLOCK_WHITE: "RLE_BLOCK_WHITE",
    CMD_DELTA_FRAME_BIT_RAW: "DELTA_FRAME_BIT (raw)",
    CMD_DELTA_FRAME_BIT_RLE: "DELTA_FRAME_BIT (RLE)",
    CMD_MASTER_FRAME: "MASTER_FRAME",
    CMD_BLOCK_FILL_BLACK: "BLOCK_FILL_BLACK",
    CMD_BLOCK_FILL_WHITE: "BLOCK_FILL_WHITE",
    CMD_BLOCK_INVERT: "BLOCK_INVERT",
    CMD_SKIP_BLOCK: "SKIP_BLOCK",
    CMD_FOR: "FOR",
    CMD_NOP: "NOP"
}

# 共通
def get_image_blocks(img_path: Path):
    img = Image.open(img_path).convert("L")
    arr = np.array(img)
    arr = (arr >= 128).astype(np.uint8)
    
    height, width = arr.shape
    if width % BLOCK_SIZE != 0 or height % BLOCK_SIZE != 0:
        raise ValueError(f"サイズエラー: {width}x{height}")

    blocks_x = width // BLOCK_SIZE
    blocks_y = height // BLOCK_SIZE

    blocks = []
    for by in range(blocks_y):
        for bx in range(blocks_x):
            block = arr[by*BLOCK_SIZE:(by+1)*BLOCK_SIZE, bx*BLOCK_SIZE:(bx+1)*BLOCK_SIZE]
            byte_list = []
            for row in block:
                val = 0
                for px in row:
                    val = (val << 1) | px
                byte_list.append(val)
            blocks.append(bytes(byte_list))
    return blocks, blocks_x, blocks_y, width, height

def parallel_load_frames(paths):
    with mp.Pool(mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(get_image_blocks, paths), total=len(paths), desc="Loading (multi-core)"))
    return results

def bits_from_bytes(b: bytes) -> list[int]:
    bits = []
    for byte in b:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits

def bytes_from_bits(bits: list[int]) -> bytes:
    ba = bytearray()
    for i in range(0, 64, 8):
        val = 0
        for j in range(8):
            val = (val << 1) | bits[i + j]
        ba.append(val)
    return ba

def try_rle_encode(bits: list[int]) -> bytes | None:
    runs = []
    if not bits:
        return None
    prev = bits[0]
    cnt = 1
    for bit in bits[1:]:
        if bit == prev:
            cnt += 1
        else:
            runs.append((prev, cnt))
            prev = bit
            cnt = 1
    runs.append((prev, cnt))

    out = bytearray()
    for val, length in runs:
        while length > 0:
            take = min(length, 127)
            out.append((val << 7) | take)
            length -= take

    if sum(r[1] for r in runs) != 64 or len(out) >= BLOCK_BYTES:
        return None
    return out

def xor_bytes(p: bytes, c: bytes) -> bytes:
    return bytes(a ^ b for a, b in zip(p, c))

# エンコード
def encode_frame(curr_blocks: list[bytes], prev_blocks: list[bytes] | None, blocks_x: int, blocks_y: int, cmd_counter: Counter) -> bytes:
    total_blocks = blocks_x * blocks_y
    stream = bytearray()
    i = 0

    if prev_blocks is None:
        stream.append(CMD_MASTER_FRAME)
        cmd_counter[CMD_MASTER_FRAME] += 1
        for block in curr_blocks:
            stream.extend(block)
        return stream

    delta_stream = bytearray()
    while i < total_blocks:
        skip_n = 0
        while i + skip_n < total_blocks and skip_n < 15 and curr_blocks[i + skip_n] == prev_blocks[i + skip_n]:
            skip_n += 1
        if skip_n > 0:
            delta_stream.append(CMD_SKIP_BLOCK | (skip_n - 1))
            cmd_counter[CMD_SKIP_BLOCK] += 1
            i += skip_n
            continue

        pattern, count = find_repeat_pattern(curr_blocks, prev_blocks, i, total_blocks)
        if count > 1:
            delta_stream.append(CMD_FOR | (count - 1))
            cmd_counter[CMD_FOR] += 1
            delta_stream.extend(pattern)
            i += count
            continue

        curr = curr_blocks[i]
        prev = prev_blocks[i]
        if all(x == 0x00 for x in curr):
            delta_stream.append(CMD_BLOCK_FILL_BLACK)
            cmd_counter[CMD_BLOCK_FILL_BLACK] += 1
        elif all(x == 0xFF for x in curr):
            delta_stream.append(CMD_BLOCK_FILL_WHITE)
            cmd_counter[CMD_BLOCK_FILL_WHITE] += 1
        elif all(a ^ b == 0xFF for a, b in zip(prev, curr)):
            delta_stream.append(CMD_BLOCK_INVERT)
            cmd_counter[CMD_BLOCK_INVERT] += 1
        else:
            bits = bits_from_bytes(curr)
            rle = try_rle_encode(bits)
            if rle:
                cmd = CMD_RLE_BLOCK_BLACK if bits[0] == 0 else CMD_RLE_BLOCK_WHITE
                delta_stream.append(cmd)
                cmd_counter[cmd] += 1
                delta_stream.extend(rle)
            else:
                xor_data = xor_bytes(prev, curr)
                xor_bits = bits_from_bytes(xor_data)
                rle_delta = try_rle_encode(xor_bits)
                if rle_delta:
                    delta_stream.append(CMD_DELTA_FRAME_BIT_RLE)
                    cmd_counter[CMD_DELTA_FRAME_BIT_RLE] += 1
                    delta_stream.extend(rle_delta)
                else:
                    delta_stream.append(CMD_MASTER_BLOCK)
                    cmd_counter[CMD_MASTER_BLOCK] += 1
                    delta_stream.extend(curr)
        i += 1

    raw_size = total_blocks * BLOCK_BYTES + 1
    if len(delta_stream) >= raw_size - 1:
        stream.append(CMD_MASTER_FRAME)
        cmd_counter[CMD_MASTER_FRAME] += 1
        for block in curr_blocks:
            stream.extend(block)
    else:
        stream.extend(delta_stream)

    return stream

def find_repeat_pattern(curr_blocks, prev_blocks, start, total):
    if start >= total:
        return b'', 0

    temp = bytearray()
    curr = curr_blocks[start]
    prev = prev_blocks[start]
    if all(x == 0x00 for x in curr):
        temp.append(CMD_BLOCK_FILL_BLACK)
    elif all(x == 0xFF for x in curr):
        temp.append(CMD_BLOCK_FILL_WHITE)
    elif all(a ^ b == 0xFF for a, b in zip(prev, curr)):
        temp.append(CMD_BLOCK_INVERT)
    else:
        bits = bits_from_bytes(curr)
        rle = try_rle_encode(bits)
        if rle:
            cmd = CMD_RLE_BLOCK_BLACK if bits[0] == 0 else CMD_RLE_BLOCK_WHITE
            temp.append(cmd)
            temp.extend(rle)
        else:
            xor_data = xor_bytes(prev, curr)
            xor_bits = bits_from_bytes(xor_data)
            rle_delta = try_rle_encode(xor_bits)
            if rle_delta:
                temp.append(CMD_DELTA_FRAME_BIT_RLE)
                temp.extend(rle_delta)
            else:
                temp.append(CMD_MASTER_BLOCK)
                temp.extend(curr)

    count = 1
    for j in range(1, 16):
        if start + j >= total:
            break
        temp2 = bytearray()
        curr2 = curr_blocks[start + j]
        prev2 = prev_blocks[start + j]
        if all(x == 0x00 for x in curr2):
            temp2.append(CMD_BLOCK_FILL_BLACK)
        elif all(x == 0xFF for x in curr2):
            temp2.append(CMD_BLOCK_FILL_WHITE)
        elif all(a ^ b == 0xFF for a, b in zip(prev2, curr2)):
            temp2.append(CMD_BLOCK_INVERT)
        else:
            bits2 = bits_from_bytes(curr2)
            rle2 = try_rle_encode(bits2)
            if rle2:
                cmd2 = CMD_RLE_BLOCK_BLACK if bits2[0] == 0 else CMD_RLE_BLOCK_WHITE
                temp2.append(cmd2)
                temp2.extend(rle2)
            else:
                xor_data2 = xor_bytes(prev2, curr2)
                xor_bits2 = bits_from_bytes(xor_data2)
                rle_delta2 = try_rle_encode(xor_bits2)
                if rle_delta2:
                    temp2.append(CMD_DELTA_FRAME_BIT_RLE)
                    temp2.extend(rle_delta2)
                else:
                    temp2.append(CMD_MASTER_BLOCK)
                    temp2.extend(curr2)
        if temp == temp2:
            count += 1
        else:
            break

    if count > 1:
        return temp, count
    return b'', 0

# エンコードメイン
def encode_main(args):
    folder = Path(args.path)
    prefix = args.name
    start_n = int(args.start)
    end_n = int(args.end)
    valid_paths = [folder / f"{prefix}{num:04d}.bmp" for num in range(start_n, end_n + 1) if (folder / f"{prefix}{num:04d}.bmp").is_file()]

    if not valid_paths:
        sys.exit("No frames")

    frame_data = parallel_load_frames(valid_paths)

    first_blocks, blocks_x, blocks_y, width, height = frame_data[0]

    for data in frame_data[1:]:
        if data[1] != blocks_x or data[2] != blocks_y:
            sys.exit("サイズ不一致")

    header = bytearray()
    header_size = 4
    header.extend(header_size.to_bytes(2, "little"))
    header.append(blocks_x)
    header.append(blocks_y)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prev_blocks = None
    global_cmd_counter = Counter()

    with open(out_path, "wb") as f:
        f.write(header)

        for idx, (curr_blocks, _, _, _, _) in enumerate(tqdm(frame_data, desc="Encoding")):
            frame_counter = Counter()
            data = encode_frame(curr_blocks, prev_blocks, blocks_x, blocks_y, frame_counter)

            # 検証
            decoded_blocks, _ = decode_frame(data, 0, prev_blocks, blocks_x, blocks_y)
            if curr_blocks != decoded_blocks:
                sys.exit(f"Frame {idx+1} verification failed")

            f.write(bytes([CMD_START_FRAME]))
            f.write(data)

            prev_blocks = curr_blocks
            global_cmd_counter.update(frame_counter)

    print("\nCommand Summary:")
    for cmd, count in sorted(global_cmd_counter.items()):
        name = CMD_NAMES.get(cmd, f"Unknown 0x{cmd:02X}")
        print(f"{name}: {count}")

# デコード
def decode_frame(stream: bytes, pos: int, prev_blocks: list[bytes] | None, blocks_x: int, blocks_y: int) -> tuple[list[bytes], int]:
    total_blocks = blocks_x * blocks_y
    curr_blocks = [b'\x00'*BLOCK_BYTES for _ in range(total_blocks)]
    if prev_blocks is None:
        prev_blocks = curr_blocks.copy()

    i_block = 0
    for_stack = []  # (remain, loop_pos)

    while i_block < total_blocks and pos < len(stream):
        cmd = stream[pos]
        pos += 1

        base = cmd & 0xF0

        if base == CMD_FOR:
            remain = (cmd & 0x0F) + 1
            loop_pos = pos
            for_stack.append((remain, loop_pos))
            # 初回実行はループ内
            continue

        # 命令実行
        if base == CMD_SKIP_BLOCK:
            skip = (cmd & 0x0F) + 1
            for _ in range(skip):
                if i_block >= total_blocks:
                    break
                curr_blocks[i_block] = prev_blocks[i_block]
                i_block += 1
        elif cmd == CMD_BLOCK_FILL_BLACK:
            curr_blocks[i_block] = b'\x00'*BLOCK_BYTES
            i_block += 1
        elif cmd == CMD_BLOCK_FILL_WHITE:
            curr_blocks[i_block] = b'\xff'*BLOCK_BYTES
            i_block += 1
        elif cmd == CMD_BLOCK_INVERT:
            curr_blocks[i_block] = bytes(~b & 0xff for b in prev_blocks[i_block])
            i_block += 1
        elif cmd == CMD_MASTER_BLOCK:
            curr_blocks[i_block] = stream[pos:pos+8]
            pos += 8
            i_block += 1
        elif cmd in (CMD_RLE_BLOCK_BLACK, CMD_RLE_BLOCK_WHITE):
            bits = []
            total_bits = 0
            while total_bits < 64 and pos < len(stream):
                b = stream[pos]
                pos += 1
                val = (b >> 7) & 1
                run = b & 0x7F
                if run == 0:
                    raise ValueError("Invalid RLE")
                bits.extend([val] * run)
                total_bits += run
            if total_bits != 64:
                raise ValueError("RLE bits !=64")
            curr_blocks[i_block] = bytes_from_bits(bits)
            i_block += 1
        elif cmd == CMD_DELTA_FRAME_BIT_RAW:
            delta = stream[pos:pos+8]
            pos += 8
            curr_blocks[i_block] = xor_bytes(prev_blocks[i_block], delta)
            i_block += 1
        elif cmd == CMD_DELTA_FRAME_BIT_RLE:
            bits = []
            total_bits = 0
            while total_bits < 64 and pos < len(stream):
                b = stream[pos]
                pos += 1
                val = (b >> 7) & 1
                run = b & 0x7F
                if run == 0:
                    raise ValueError("Invalid DELTA RLE")
                bits.extend([val] * run)
                total_bits += run
            if total_bits != 64:
                raise ValueError("DELTA RLE bits !=64")
            delta = bytes_from_bits(bits)
            curr_blocks[i_block] = xor_bytes(prev_blocks[i_block], delta)
            i_block += 1
        elif cmd == CMD_MASTER_FRAME:
            for j in range(total_blocks - i_block):
                curr_blocks[i_block + j] = stream[pos:pos+8]
                pos += 8
            i_block = total_blocks
        elif base == CMD_NOP:
            pass
        else:
            raise ValueError(f"Unknown cmd 0x{cmd:02X}")

        # FOR処理 (実行後)
        if for_stack:
            remain, loop_pos = for_stack[-1]
            remain -= 1
            if remain > 0:
                for_stack[-1] = (remain, loop_pos)
                pos = loop_pos  # 再実行
            else:
                for_stack.pop()

    if i_block < total_blocks:
        raise ValueError("Incomplete frame")

    return curr_blocks, pos

# デコードメイン
def decode_main(args):
    in_path = Path(args.input)
    with open(in_path, "rb") as f:
        data = f.read()

    pos = 0
    header_size = int.from_bytes(data[pos:pos+2], "little")
    pos += 2
    blocks_x = data[pos]
    pos += 1
    blocks_y = data[pos]
    pos += 1

    width = blocks_x * BLOCK_SIZE
    height = blocks_y * BLOCK_SIZE

    out_dir = Path(args.path)
    out_dir.mkdir(parents=True, exist_ok=True)

    prev_blocks = None
    fn = 1
    while pos < len(data):
        if data[pos] != CMD_START_FRAME:
            break
        pos += 1

        curr_blocks, consumed = decode_frame(data[pos:], 0, prev_blocks, blocks_x, blocks_y)
        pos += consumed

        arr = np.zeros((height, width), dtype=np.uint8)
        for by in range(blocks_y):
            for bx in range(blocks_x):
                block_bytes = curr_blocks[by * blocks_x + bx]
                for r in range(8):
                    byte = block_bytes[r]
                    for c in range(8):
                        bit = (byte >> (7 - c)) & 1
                        arr[by*8 + r, bx*8 + c] = 255 if bit else 0

        img = Image.fromarray(arr, mode='L')
        img.save(out_dir / f"{args.name}{fn:04d}.bmp")
        prev_blocks = curr_blocks
        fn += 1

    print(f"Decoded {fn-1} frames")

# メイン
def main():
    parser = argparse.ArgumentParser(description="BadCodec")
    parser.add_argument("-c", "--command", required=True, choices=["e", "d"])
    parser.add_argument("-i", "--input", help="入力")
    parser.add_argument("-o", "--output", help="出力")
    parser.add_argument("-n", "--name", default="frame_")
    parser.add_argument("-s", "--start", default="1")
    parser.add_argument("-e", "--end", required=True)
    parser.add_argument("-p", "--path", required=True)
    args = parser.parse_args()

    if args.command == "e":
        encode_main(args)
    elif args.command == "d":
        decode_main(args)

if __name__ == "__main__":
    main()
    
    