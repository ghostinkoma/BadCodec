import os
import sys
import time
import argparse
import numpy as np
from PIL import Image
import struct
from collections import Counter

# --- ANSI Colors for Console Output ---
C_SKIP  = "\033[90m" # Gray
C_FILL  = "\033[97m" # White
C_INV   = "\033[96m" # Cyan
C_SHIFT = "\033[94m" # Blue
C_RLE   = "\033[92m" # Green
C_MAST  = "\033[91m" # Red
C_RESET = "\033[0m"

class BadCodec:
    def __init__(self, width=0, height=0, block_size=8):
        self.w, self.h = width, height
        self.bs = block_size
        if width > 0:
            self.bx, self.by = width // block_size, height // block_size
            self.total_blocks = self.bx * self.by
        
        # 統計用カウンタ
        self.stats = Counter()
        self.total_bytes = 0
        self.raw_bytes = 0

        # RLE(Block)用スキャンパターン (Vertical/Horizontal)
        self.rle_patterns = [
            (False, 0, 0, 1, 1), (False, 7, 0, -1, 1),
            (False, 0, 7, 1, -1), (False, 7, 7, -1, -1),
            (True,  0, 0, 1, 1), (True,  7, 0, -1, 1),
            (True,  0, 7, 1, -1), (True,  7, 7, -1, -1)
        ]

    # ---------------------------------------------------------
    # Helper: Scan Coordinates for Block RLE
    # ---------------------------------------------------------
    def _get_scan_coords(self, p_idx):
        is_vert, sx, sy, dx, dy = self.rle_patterns[p_idx]
        coords = []
        if not is_vert: # Horizontal priority
            for i in range(8): # Y
                y = sy + i * dy
                for j in range(8): # X
                    x = sx + j * dx
                    coords.append((x, y))
        else: # Vertical priority
            for j in range(8): # X
                x = sx + j * dx
                for i in range(8): # Y
                    y = sy + i * dy
                    coords.append((x, y))
        return coords

    # ---------------------------------------------------------
    # Helper: Shift Logic (X/Y Separated, Signed)
    # ---------------------------------------------------------
    def _apply_shift(self, prev_b, axis, sign, amount):
        """
        axis: 0=X, 1=Y
        sign: 0=+, 1=-
        amount: 1-7 (pixels)
        """
        shifted = np.zeros((8, 8), dtype=np.uint8)
        # sign=1 (-) means shift left/up (idx decreases), sign=0 (+) means right/down (idx increases)
        # Shift value for numpy roll/slicing
        # Logic: If we shift image RIGHT (+x), src pixels come from left.
        # Here we simulate the *result* of the move.
        # If block moves +X (right), empty space appears on left.
        
        val = -amount if sign == 1 else amount
        
        src_y_s, src_y_e = 0, 8
        src_x_s, src_x_e = 0, 8
        dst_y_s, dst_y_e = 0, 8
        dst_x_s, dst_x_e = 0, 8

        if axis == 0: # X-Shift
            dx = val
            # If dx > 0 (Right): dst starts at dx, src starts at 0. len = 8-dx
            # If dx < 0 (Left):  dst starts at 0, src starts at -dx. len = 8+dx
            src_x_s = max(0, -dx); src_x_e = min(8, 8 - dx)
            dst_x_s = max(0, dx);  dst_x_e = min(8, 8 + dx)
        else: # Y-Shift
            dy = val
            src_y_s = max(0, -dy); src_y_e = min(8, 8 - dy)
            dst_y_s = max(0, dy);  dst_y_e = min(8, 8 + dy)
            
        if dst_y_e > dst_y_s and dst_x_e > dst_x_s:
            shifted[dst_y_s:dst_y_e, dst_x_s:dst_x_e] = \
                prev_b[src_y_s:src_y_e, src_x_s:src_x_e]
        return shifted

    # ---------------------------------------------------------
    # Block RLE Packer (for Block Stream)
    # ---------------------------------------------------------
    def _pack_rle_block(self, block, p_idx, inv):
        coords = self._get_scan_coords(p_idx)
        seq = [block[y, x] for x, y in coords]
        runs, curr_color, count = [], inv, 0
        for p in seq:
            if p == curr_color:
                count += 1
                if count == 63: runs.extend([63, 0]); count = 0
            else:
                runs.append(count); count = 1; curr_color = 1 - curr_color
        runs.append(count)
        
        packed = bytearray()
        for i in range(0, len(runs), 4):
            c = (runs[i:i+4] + [0]*4)[:4]
            b1 = (c[0] & 0x3F) | ((c[1] & 0x03) << 6)
            b2 = ((c[1] >> 2) & 0x0F) | ((c[2] & 0x0F) << 4)
            b3 = ((c[2] >> 4) & 0x03) | ((c[3] & 0x3F) << 2)
            packed.extend([b1, b2, b3])
        return bytes(packed)

    def _unpack_rle_block(self, data_stream, p_idx, inv):
        coords = self._get_scan_coords(p_idx)
        block = np.zeros((8, 8), dtype=np.uint8)
        runs, total_px = [], 0
        while total_px < 64:
            b = data_stream.read(3)
            if not b: break
            r = [b[0] & 0x3F, ((b[0] >> 6) & 0x03) | ((b[1] & 0x0F) << 2),
                 ((b[1] >> 4) & 0x0F) | ((b[2] & 0x03) << 4), (b[2] >> 2) & 0x3F]
            for val in r:
                runs.append(val); total_px += val
                if total_px >= 64: break
        
        curr_color, px_ptr = inv, 0
        for r in runs:
            for _ in range(r):
                if px_ptr < 64:
                    x, y = coords[px_ptr]; block[y, x] = curr_color; px_ptr += 1
            curr_color = 1 - curr_color
        return block

    # ---------------------------------------------------------
    # Logic 1: Block Stream Encoder (The detailed worker)
    # ---------------------------------------------------------
    def _generate_block_stream(self, frame, prev_frame):
        cmds = []
        block_types = [] # For UI visualization
        b_idx = 0
        
        # Pre-calculate counts for summary later (temporary)
        local_stats = Counter()

        while b_idx < self.total_blocks:
            by, bx = divmod(b_idx, self.bx)
            curr_b = frame[by*8:(by+1)*8, bx*8:(bx+1)*8]
            prev_b = prev_frame[by*8:(by+1)*8, bx*8:(bx+1)*8] if prev_frame is not None else np.zeros((8,8), dtype=np.uint8)

            # --- 1. SKIP (Priority: Top) ---
            if np.array_equal(curr_b, prev_b):
                cnt = 1
                while b_idx + cnt < self.total_blocks and cnt < 64:
                    ny, nx = divmod(b_idx + cnt, self.bx)
                    p_curr = frame[ny*8:(ny+1)*8, nx*8:(nx+1)*8]
                    p_prev = prev_frame[ny*8:(ny+1)*8, nx*8:(nx+1)*8] if prev_frame is not None else np.zeros((8,8), dtype=np.uint8)
                    if np.array_equal(p_curr, p_prev): cnt += 1
                    else: break
                
                # Opcode: 10xxxxxx (0x80 | count-1)
                cmds.append(0x80 | (cnt - 1))
                block_types.extend(['S'] * cnt)
                b_idx += cnt
                local_stats['SKIP_BLOCK'] += 1
                continue

            # --- 2. INVERT (Priority: High) ---
            inv_prev_b = 1 - prev_b
            if np.array_equal(curr_b, inv_prev_b):
                cnt = 1
                while b_idx + cnt < self.total_blocks and cnt < 32:
                    ny, nx = divmod(b_idx + cnt, self.bx)
                    p_curr = frame[ny*8:(ny+1)*8, nx*8:(nx+1)*8]
                    p_prev = prev_frame[ny*8:(ny+1)*8, nx*8:(nx+1)*8] if prev_frame is not None else np.zeros((8,8), dtype=np.uint8)
                    if np.array_equal(p_curr, 1 - p_prev): cnt += 1
                    else: break
                
                # Opcode: 000xxxxx (0x00 | count-1)
                cmds.append(0x00 | (cnt - 1))
                block_types.extend(['I'] * cnt)
                b_idx += cnt
                local_stats['BLOCK_INVERT'] += 1
                continue

            # --- 3. FILL (Priority: High) ---
            if np.all(curr_b == 0) or np.all(curr_b == 1):
                val = 1 if np.all(curr_b == 1) else 0
                is_black = (val == 1) # Assign: 1=Black(ON), 0=White(OFF) as per typical bad apple logic
                
                cnt = 1
                while b_idx + cnt < self.total_blocks and cnt < 8:
                    ny, nx = divmod(b_idx + cnt, self.bx)
                    blk = frame[ny*8:(ny+1)*8, nx*8:(nx+1)*8]
                    if np.all(blk == val): cnt += 1
                    else: break
                
                # Opcode: 0010 C CCC (C=Color, CCC=Count-1)
                # Base 0x20. If Black (1), bit3=1 -> 0x28.
                cmd_byte = 0x20 | ((1 if is_black else 0) << 3) | (cnt - 1)
                cmds.append(cmd_byte)
                block_types.extend(['F'] * cnt)
                b_idx += cnt
                local_stats['FILL_BLOCK'] += 1
                continue

            # --- 4. SHIFT (Priority: Mid - High Computation) ---
            # Search X and Y, -7 to +7 (Amount 1-7, Sign 0/1)
            # Opcode: 01 SA MMMM (S=Sign, A=Axis, M=Amount shifted << 1 ? No, "bit3-1:Amount")
            # Wait, prompt says: bit6=1, bit5=Axis, bit4=Sign, bit3-1=Amount.
            # Format: 01 [Axis] [Sign] [Amt2][Amt1][Amt0] [0?]
            # Let's align: 0x40 | (Axis<<5) | (Sign<<4) | (Amount<<1)
            
            best_shift = None
            found_shift = False
            
            # X-Axis Search
            for s in [0, 1]: # 0=+, 1=-
                for amt in range(1, 8):
                    shifted = self._apply_shift(prev_b, 0, s, amt)
                    if np.array_equal(curr_b, shifted):
                        best_shift = 0x40 | (0 << 5) | (s << 4) | (amt << 1)
                        found_shift = True
                        break
                if found_shift: break
            
            if not found_shift:
                # Y-Axis Search
                for s in [0, 1]:
                    for amt in range(1, 8):
                        shifted = self._apply_shift(prev_b, 1, s, amt)
                        if np.array_equal(curr_b, shifted):
                            best_shift = 0x40 | (1 << 5) | (s << 4) | (amt << 1)
                            found_shift = True
                            break
                    if found_shift: break
            
            if found_shift:
                cmds.append(best_shift)
                block_types.append('H') # H for sHift
                b_idx += 1
                local_stats['SHIFT_BLOCK'] += 1
                continue

            # --- 5. RLE Block (Priority: Low) ---
            best_rle = (999, None, None)
            for p_idx in range(8):
                for inv in [0, 1]:
                    d = self._pack_rle_block(curr_b, p_idx, inv)
                    # Cost: 1 byte (Opcode) + len(d)
                    if len(d) + 1 < best_rle[0]:
                        # Opcode: 0001 P PPI (P=Pattern, I=Inv) -> 0x10 base
                        op = 0x10 | (p_idx << 1) | inv
                        best_rle = (len(d) + 1, op, d)
            
            # --- 6. Master Block (Fallback) ---
            # Cost: 1 byte (Opcode) + 8 bytes (Data) = 9 bytes
            # Opcode: 0011 0100 (0x34)
            master_cost = 9
            
            if best_rle[0] < master_cost:
                cmds.append(best_rle[1])
                cmds.append(best_rle[2])
                block_types.append('R')
                local_stats['RLE_BLOCK'] += 1
            else:
                cmds.append(0x34)
                cmds.append(np.packbits(curr_b.flatten(), bitorder='little').tobytes())
                block_types.append('M')
                local_stats['MASTER_BLOCK'] += 1
            
            b_idx += 1

        return cmds, block_types, local_stats

    # ---------------------------------------------------------
    # Logic 2: RLE Frame Encoder (0x35)
    # ---------------------------------------------------------
    def _generate_rle_frame(self, frame):
        flat = frame.flatten()
        data = bytearray()
        
        if len(flat) == 0: return b'\x00'
        
        curr_clr = flat[0]
        run = 0
        
        for p in flat:
            if p == curr_clr:
                run += 1
            else:
                # Flush run
                while run > 0:
                    if run > 127:
                        # Emit max chunk, then special 0xFF to continue same color
                        chunk = 127
                        # Byte: [Color 1bit] [Run 7bit]
                        b = (curr_clr << 7) | chunk
                        data.append(b)
                        run -= chunk
                        
                        # Emit Extension markers if run still remains
                        while run > 127:
                            data.append(0xFF) # "Keep current color, add 127"
                            run -= 127
                        
                        # At this point run <= 127. 
                        # We need to emit the remainder.
                        # Important: The loop continues to next pixel which is DIFF color.
                        # But we just finished the massive block of SAME color.
                        # We must emit the remainder of SAME color.
                        # If we emit (curr_clr << 7) | run, that's fine.
                        if run > 0:
                            b = (curr_clr << 7) | run
                            data.append(b)
                            run = 0
                    else:
                        b = (curr_clr << 7) | run
                        data.append(b)
                        run = 0
                
                curr_clr = p
                run = 1
        
        # Final flush
        while run > 0:
            if run > 127:
                chunk = 127
                data.append((curr_clr << 7) | chunk)
                run -= chunk
                while run > 127:
                    data.append(0xFF)
                    run -= 127
            if run > 0:
                data.append((curr_clr << 7) | run)
                run = 0
                
        data.append(0x00) # Terminator
        return bytes(data)

    def _decode_rle_frame(self, f_stream):
        pixels = []
        target_px = self.w * self.h
        curr_color = 0 # Default, updated by first byte
        
        while len(pixels) < target_px:
            b = f_stream.read(1)
            if not b: break
            val = b[0]
            
            if val == 0x00: break # Terminator
            
            if val == 0xFF:
                # Extension: Draw 127 pixels of CURRENT color
                pixels.extend([curr_color] * 127)
            else:
                curr_color = (val >> 7) & 1
                run = val & 0x7F
                pixels.extend([curr_color] * run)
        
        arr = np.array(pixels[:target_px], dtype=np.uint8)
        return arr.reshape((self.h, self.w))

    # ---------------------------------------------------------
    # Logic 3: Master Frame (0x36)
    # ---------------------------------------------------------
    def _generate_master_frame(self, frame):
        return np.packbits(frame.flatten(), bitorder='little').tobytes()

    # ---------------------------------------------------------
    # Main Encoder: The Triangle Comparison
    # ---------------------------------------------------------
    def encode_frame(self, frame, prev_frame):
        # -- 0. Global Frame Shortcuts (Topmost Tree) --
        # These are usually always smallest if applicable
        if prev_frame is not None:
            if np.array_equal(frame, prev_frame):
                self.stats['SKIP_FRAME'] += 1
                return [0x30, 0x01], ['S']*self.total_blocks # 0x30=Delimiter, 0x01=SKIP_FRAME (Control)
            if np.array_equal(frame, 1 - prev_frame):
                self.stats['INVERT_PREV_FRAME'] += 1
                return [0x30, 0x07], ['I']*self.total_blocks

        if np.all(frame == 1):
            self.stats['FRAME_FILL_BLACK'] += 1
            return [0x30, 0x02], ['F']*self.total_blocks
        if np.all(frame == 0):
            self.stats['FRAME_FILL_WHITE'] += 1
            return [0x30, 0x03], ['F']*self.total_blocks

        # -- Competition Logic --
        
        # 1. Block Stream
        # Delimiter (1) + Commands
        blk_cmds, blk_types, blk_stats = self._generate_block_stream(frame, prev_frame)
        size_block = 1 # 0x30 Delimiter
        for c in blk_cmds:
            if isinstance(c, int): size_block += 1
            else: size_block += len(c)
        
        # 2. RLE Frame
        # Delimiter (1) + Opcode (1) + Data
        rle_data = self._generate_rle_frame(frame)
        size_rle = 1 + 1 + len(rle_data) # 0x30 + 0x05 + Data
        
        # 3. Master Frame
        # Delimiter (1) + Opcode (1) + Data
        mast_data = self._generate_master_frame(frame)
        size_master = 1 + 1 + len(mast_data) # 0x30 + 0x06 + Data

        # Comparison ①: Block vs RLE
        # "RLE_FRAMEとの比較で①より小さい場合のみRLE_FRAMEを...採用"
        # Logic: if Size_Block <= Size_RLE: candidate = Block; else: candidate = RLE
        
        winner = None
        
        if size_block <= size_rle:
            candidate_size = size_block
            candidate_type = 'BLOCK'
        else:
            candidate_size = size_rle
            candidate_type = 'RLE'
            
        # Comparison ②: Candidate vs Master
        # "MASTE_FRAMEと②をくらべてなおMASTE_FRAMEの方が小さいばあい...採用"
        
        if candidate_size <= size_master:
            final_type = candidate_type
        else:
            final_type = 'MASTER'
            
        # -- Final Emit --
        final_cmds = [0x30] # Frame Delimiter
        final_ui = []
        
        if final_type == 'BLOCK':
            final_cmds.extend(blk_cmds)
            final_ui = blk_types
            # Merge stats
            self.stats.update(blk_stats)
            
        elif final_type == 'RLE':
            # Opcode 0x05 (Frame Control) inside Frame Delimiter context? 
            # Spec says "FRAME_CONTROL... 0101 : RLE_FRAME"
            # Command structure: 0x30 (Delimiter) -> 0x35 (RLE_FRAME) -> Data
            # Wait, 0x30 IS Frame Delimiter.
            # Frame Control sub-commands: 0x00=Delimiter, 0x01=Skip... 0x05=RLE.
            # But opcode bits: 0011 xxxx.
            # 0x30 is Delimiter.
            # 0x35 is RLE_FRAME.
            # So we emit [0x30, 0x35, Data...].
            # Actually, standard flow is: 0x30 (New Frame Start) -> Commands.
            # If Command is 0x35, it handles the whole frame.
            final_cmds.append(0x35)
            final_cmds.append(rle_data)
            final_ui = ['R'] * self.total_blocks
            self.stats['RLE_FRAME'] += 1
            
        elif final_type == 'MASTER':
            final_cmds.append(0x36)
            final_cmds.append(mast_data)
            final_ui = ['M'] * self.total_blocks
            self.stats['MASTER_FRAME'] += 1
            
        return final_cmds, final_ui


def update_ui(curr, total, start_time, block_types, bx, mode):
    elapsed = time.time() - start_time
    sys.stdout.write('\033[H')
    print(f"{C_RESET}BadCodec v2.1 | Strict Logic & Shift")
    print(f"Mode: {mode}")
    print(f"Frame: {curr+1:04d}/{total:04d} | Time: {elapsed:6.1f}s")
    prog = int(40 * (curr + 1) / total)
    print(f"Progress: [{'#'*prog}{'.'*(40-prog)}] {int(100*(curr+1)/total)}%")
    
    if block_types:
        print(f"\nBlock Map (Top 16 rows):")
        cmap = {'S':C_SKIP, 'F':C_FILL, 'I':C_INV, 'H':C_SHIFT, 'M':C_MAST, 'R':C_RLE}
        for r in range(min(16, len(block_types)//bx)):
            row = block_types[r*bx : (r+1)*bx]
            print(" ".join([f"{cmap.get(c, C_RESET)}{c}{C_RESET}" for c in row]))
    sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', choices=['e', 'd'], required=True, help="e=Encode, d=Decode")
    parser.add_argument('-i', '--input', help='Input .bad file')
    parser.add_argument('-p', '--path', required=True, help='BMP Folder Path')
    parser.add_argument('-n', '--suffix', default='frame_')
    parser.add_argument('-s', '--start', type=int, default=1)
    parser.add_argument('-e', '--end', type=int)
    parser.add_argument('-o', '--output', default='output.bad')
    args = parser.parse_args()

    if args.t == 'e':
        if args.end is None: return print("Error: -e (end frame) required for encoding.")
        
        # Init Codec
        f0_path = os.path.join(args.path, f"{args.suffix}{args.start:04d}.bmp")
        if not os.path.exists(f0_path): return print(f"File not found: {f0_path}")
        f0 = Image.open(f0_path).convert('1')
        codec = BadCodec(f0.width, f0.height, 8)
        
        # Header: 'Bad' + Ver(3) + W + H + BS
        header = b'Bad' + struct.pack('<H H H B', 3, f0.width, f0.height, 8)
        full_data = bytearray(header)
        
        codec.raw_bytes = (args.end - args.start + 1) * (f0.width * f0.height // 8)
        
        prev_f = None
        st_time = time.time()
        print("\033[2J\033[H", end="")
        
        for idx, i in enumerate(range(args.start, args.end + 1)):
            fpath = os.path.join(args.path, f"{args.suffix}{i:04d}.bmp")
            if not os.path.exists(fpath): continue
            
            frame = np.array(Image.open(fpath).convert('1'), dtype=np.uint8)
            
            cmds, bt = codec.encode_frame(frame, prev_f)
            
            for c in cmds:
                if isinstance(c, int): full_data.append(c)
                else: full_data.extend(c)
            
            prev_f = frame.copy()
            update_ui(idx, args.end - args.start + 1, st_time, bt, codec.bx, "ENCODING")
            
        with open(args.output, 'wb') as f: f.write(full_data)
        codec.total_bytes = len(full_data)
        
    else: # Decode
        if not args.input: return print("Error: -i required for decoding.")
        with open(args.input, 'rb') as f:
            if f.read(3) != b'Bad': return print("Invalid Header")
            ver, w, h, bs = struct.unpack('<H H H B', f.read(7))
            codec = BadCodec(w, h, bs)
            
            if not os.path.exists(args.path): os.makedirs(args.path)
            
            curr_f = np.zeros((h, w), dtype=np.uint8)
            prev_f = np.zeros((h, w), dtype=np.uint8)
            
            f.seek(0, 2); total_sz = f.tell(); f.seek(10)
            st_time = time.time()
            f_idx = 0
            
            print("\033[2J\033[H", end="")
            
            while f.tell() < total_sz:
                op = f.read(1)
                if not op: break
                val = op[0]
                
                # Check 0x30 Delimiter/Control Context
                if val == 0x30:
                    # Next byte might be Frame Control or Block Stream
                    # Look ahead or just enter loop?
                    # The spec implies 0x30 is delimiter. The *next* command decides frame type.
                    # Or 0x30 allows a sequence of Block commands.
                    pass 
                
                # Handle Frame Control Commands (0x3X)
                # Note: Block Stream commands (0x80, 0x40...) will appear after 0x30
                
                # We need to detect "Single Frame Command" vs "Block Loop"
                # Peek or just Read next?
                # If we are strictly following "0x30 = Delimiter", then loop starts.
                
                # Let's simplify: 
                # If next is 0x35 (RLE) or 0x36 (Master) or 0x31 (Skip)... execute and finish frame.
                # Else: Block Loop.
                
                next_b = f.read(1)
                if not next_b: break
                cmd = next_b[0]
                
                frame_done = False
                
                # Frame Control Group
                if (cmd & 0xF0) == 0x30 and (cmd & 0x08) == 0: 
                    sub = cmd & 0x07
                    if sub == 0x01: # SKIP FRAME
                        codec.stats['SKIP_FRAME'] += 1; frame_done = True
                    elif sub == 0x02: # FILL BLACK
                        curr_f.fill(1); codec.stats['FRAME_FILL_BLACK'] += 1; frame_done = True
                    elif sub == 0x03: # FILL WHITE
                        curr_f.fill(0); codec.stats['FRAME_FILL_WHITE'] += 1; frame_done = True
                    elif sub == 0x05: # RLE FRAME
                        curr_f = codec._decode_rle_frame(f)
                        codec.stats['RLE_FRAME'] += 1; frame_done = True
                    elif sub == 0x06: # MASTER FRAME
                        raw = f.read(w*h//8)
                        curr_f = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder='little').reshape(h, w)
                        codec.stats['MASTER_FRAME'] += 1; frame_done = True
                    elif sub == 0x07: # INVERT PREV
                        curr_f = 1 - prev_f; codec.stats['INVERT_PREV_FRAME'] += 1; frame_done = True
                    # If 0x34 (Master Block), it falls through to Block Loop (technically 0x34 is FrameControl group in bit logic, but used in block stream)
                
                if not frame_done:
                    # It's a Block Stream. 'cmd' is the first block command.
                    # We must process 'cmd' then loop until block count filled.
                    b_idx = 0
                    
                    # Hack: Put back 'cmd' to process in generic loop? 
                    # Or process 'cmd' now.
                    # To keep clean, let's treat `cmd` as current command.
                    
                    active_cmd = cmd
                    while b_idx < codec.total_blocks:
                        # Process active_cmd
                        c = active_cmd
                        
                        if c & 0x80: # CONTROL (Bit7=1)
                            if (c & 0x40) == 0: # SKIP (0x80)
                                cnt = (c & 0x3F) + 1
                                # Copy prev
                                # But actually we assume curr_f is initialized to 0 or something?
                                # No, we must copy prev_f explicitly because decoder might recycle buffer 
                                # OR if we write to new buffer, we must copy.
                                # Let's assume we copy.
                                # Efficient way: Since we update curr_f incrementally,
                                # we should probably init curr_f = prev_f.copy() at start?
                                # But FILL commands overwrite.
                                # Let's copy on demand.
                                for _ in range(cnt):
                                    ny, nx = divmod(b_idx, codec.bx)
                                    curr_f[ny*8:(ny+1)*8, nx*8:(nx+1)*8] = prev_f[ny*8:(ny+1)*8, nx*8:(nx+1)*8]
                                    b_idx += 1
                                codec.stats['SKIP_BLOCK'] += 1
                            else: # FOR (0xC0) - Placeholder
                                b_idx += 1
                        else: # DRAW (Bit7=0)
                            if c & 0x40: # SHIFT (0x40)
                                # 01 SA MMMM
                                axis = (c >> 5) & 1
                                sign = (c >> 4) & 1
                                amt  = (c >> 1) & 0x07 # Bit 3-1
                                
                                by, bx = divmod(b_idx, codec.bx)
                                p_blk = prev_f[by*8:(by+1)*8, bx*8:(bx+1)*8]
                                curr_f[by*8:(by+1)*8, bx*8:(bx+1)*8] = codec._apply_shift(p_blk, axis, sign, amt)
                                b_idx += 1
                                codec.stats['SHIFT_BLOCK'] += 1
                            else:
                                if (c & 0x20) == 0: # INVERT (0x00)
                                    cnt = (c & 0x1F) + 1
                                    for _ in range(cnt):
                                        ny, nx = divmod(b_idx, codec.bx)
                                        curr_f[ny*8:(ny+1)*8, nx*8:(nx+1)*8] = 1 - prev_f[ny*8:(ny+1)*8, nx*8:(nx+1)*8]
                                        b_idx += 1
                                    codec.stats['BLOCK_INVERT'] += 1
                                else:
                                    if (c & 0x10) == 0: # FILL (0x20)
                                        is_blk = (c >> 3) & 1
                                        cnt = (c & 0x07) + 1
                                        val = 1 if is_blk else 0
                                        for _ in range(cnt):
                                            ny, nx = divmod(b_idx, codec.bx)
                                            curr_f[ny*8:(ny+1)*8, nx*8:(nx+1)*8] = val
                                            b_idx += 1
                                        codec.stats['FILL_BLOCK'] += 1
                                    elif (c & 0x08) == 0: # 0011 0xxx -> Master Block (0x34) is here
                                        if c == 0x34:
                                            raw = f.read(8)
                                            by, bx = divmod(b_idx, codec.bx)
                                            curr_f[by*8:(by+1)*8, bx*8:(bx+1)*8] = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder='little').reshape(8,8)
                                            b_idx += 1
                                            codec.stats['MASTER_BLOCK'] += 1
                                        # 0x30-0x37 Frame controls handled earlier?
                                        # But Block RLE (0x10) ?
                                        # Wait, RLE Block is 0001 xxxx -> 0x10.
                                        # (c & 0xF0) == 0x10.
                                    else:
                                        pass
                                        
                            # Special Check for RLE Block (0x10 base)
                            # The previous if/else tree missed 0x10 specifically because 0x00(Inv) and 0x20(Fill).
                            # 0x10 is between them.
                            # Re-eval check:
                            if (c & 0xF0) == 0x10:
                                by, bx = divmod(b_idx, codec.bx)
                                curr_f[by*8:(by+1)*8, bx*8:(bx+1)*8] = codec._unpack_rle_block(f, (c >> 1) & 0x07, c & 0x01)
                                b_idx += 1
                                codec.stats['RLE_BLOCK'] += 1
                        
                        if b_idx < codec.total_blocks:
                            nxt = f.read(1)
                            if not nxt: break
                            active_cmd = nxt[0]

                # End of Frame
                f_idx += 1
                Image.fromarray(curr_f * 255).convert('1').save(os.path.join(args.path, f"{args.suffix}{f_idx:04d}.bmp"))
                prev_f = curr_f.copy()
                update_ui(f.tell(), total_sz, st_time, None, codec.bx, "DECODING")

    # --- FINAL SUMMARY ---
    print(f"\n\n{C_RESET}=== BadCodec v2.1 Execution Summary ===")
    print("-" * 40)
    print(f"{'Command':<20} | {'Count':>10}")
    print("-" * 40)
    for k, v in sorted(codec.stats.items(), key=lambda x: -x[1]):
        print(f"{k:<20} | {v:10d}")
    print("-" * 40)
    
    if args.t == 'e' and codec.raw_bytes > 0:
        ratio = (codec.total_bytes / codec.raw_bytes) * 100
        print(f"Raw Size    : {codec.raw_bytes:,} bytes")
        print(f"Encoded Size: {codec.total_bytes:,} bytes")
        print(f"Compression : {ratio:.2f}% ({(100-ratio):.2f}% Reduction)")
        print(f"Target      : < 10.00%")

if __name__ == '__main__': main()