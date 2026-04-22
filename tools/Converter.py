# -*- coding: utf-8 -*-
import os
import subprocess
import sys

def run_cmd(cmd):
    print(f"\n[Running]: {cmd}")
    return subprocess.run(cmd, shell=True)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    os.chdir(root_dir)
    py3 = f'"{sys.executable}"'

    print("===============================================")
    print("   ESP32-C3 Video & Audio Header Builder       ")
    print("      (With Compression Ratio Summary)         ")
    print("===============================================")

    url = input("\n[1] YouTube URL: ").strip()
    if not url: return

    sizes = [
        ("64x32", 64, 32),
        ("72x40", 72, 40),
        ("128x64", 128, 64),
        ("256x128", 256, 128),
        ("230x240", 230, 240)
    ]
    for i, (label, _, _) in enumerate(sizes, 1):
        print(f"  {i}: {label}")
    size_idx = int(input("Choice [3]: ") or "3") - 1
    label, width, height = sizes[size_idx]

    raw_rate = int(input("\n[2] Audio Sample Rate [18000]: ") or "18000")
    sample_rate = round(raw_rate / 500) * 500

    # パス設定
    res_dir = os.path.normpath(os.path.join(root_dir, "resouce"))
    inc_dir = os.path.normpath(os.path.join(root_dir, "include"))
    frames_dir = os.path.join(res_dir, "Frames", label)
    
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(inc_dir, exist_ok=True)

    video_mp4 = os.path.join(res_dir, "download.mp4")
    temp_bad  = os.path.join(res_dir, "output.bad")
    temp_wav  = os.path.join(res_dir, "audio.wav")
    
    codec_py  = os.path.join(script_dir, "Codec.py")
    wave_py   = os.path.join(script_dir, "Wave2AdpcmH.py")
    
    video_h_out = os.path.join(inc_dir, "bad_data.h")
    audio_h_out = os.path.join(inc_dir, "adpcm4.h")

    # --- 1. DOWNLOAD ---
    print("\n>>> [1/5] Downloading video...")
    run_cmd(f'yt-dlp -f "bestvideo[height<=480]+bestaudio/best" --merge-output-format mp4 -o "{video_mp4}" {url}')

    # --- 2. VIDEO STRETCH ---
    print(f"\n>>> [2/5] Extracting & Stretching to {width}x{height}...")
    vf = f"scale={width}:{height},format=gray,geq='lum=if(gt(lum(X,Y),128),255,0)'"
    run_cmd(f'ffmpeg -y -i "{video_mp4}" -vf "{vf}" -vcodec bmp "{frames_dir}/frame_%04d.bmp"')

    # 元のBMPの合計サイズを計算
    raw_video_size = sum(os.path.getsize(os.path.join(frames_dir, f)) for f in os.listdir(frames_dir) if f.endswith('.bmp'))

    # --- 3. ENCODE VIDEO (.bad) ---
    print(f"\n>>> [3/5] Encoding Video to .bad binary...")
    if os.path.exists(temp_bad): os.remove(temp_bad)
    run_cmd(f'{py3} "{codec_py}" -t e -p "{frames_dir}/" -n frame_ -s 0001 -e {len([f for f in os.listdir(frames_dir) if f.endswith(".bmp")]):04d} -o "{temp_bad}"')

    # --- 4. AUDIO HEADER ---
    print("\n>>> [4/5] Generating Audio Header...")
    run_cmd(f'ffmpeg -y -i "{video_mp4}" -vn -ar {sample_rate} -ac 1 -f wav "{temp_wav}"')
    raw_audio_size = os.path.getsize(temp_wav)
    run_cmd(f'{py3} "{wave_py}" -i "{temp_wav}" -o "{audio_h_out}" -r {sample_rate}')

    # --- 5. VIDEO HEADER ---
    print("\n>>> [5/5] Generating Video Header (bad_data.h)...")
    run_cmd(f'{py3} "{codec_py}" -t c -i "{temp_bad}" -H "{video_h_out}"')

    # --- 圧縮率サマリーの表示 ---
    print("\n" + "="*50)
    if os.path.exists(video_h_out) and os.path.exists(audio_h_out):
        final_video_size = os.path.getsize(temp_bad)
        # adpcm4.hのサイズは、大まかに temp_wav(16bit) の約1/4 + α (ヘッダー形式)
        final_audio_size = os.path.getsize(audio_h_out) # ソースコードとしてのサイズ

        v_ratio = (final_video_size / raw_video_size) * 100 if raw_video_size > 0 else 0
        # 音声の圧縮率は temp_wav(PCM 16bit) とのバイナリ比較が妥当
        # ADPCMは1サンプル4bitなので、論理的に25%（1/4）程度になる
        a_ratio = 25.0 

        print("   SUCCESS: Headers Generated!")
        print("-" * 50)
        print(f"   [VIDEO] {width}x{height}")
        print(f"    - Raw (BMPs) : {raw_video_size / 1024 / 1024:.2f} MB")
        print(f"    - Encoded    : {final_video_size / 1024:.2f} KB")
        print(f"    - Ratio      : {v_ratio:.2f} %")
        print("-" * 50)
        print(f"   [AUDIO] {sample_rate} Hz (ADPCM 4bit)")
        print(f"    - Raw (WAV)  : {raw_audio_size / 1024 / 1024:.2f} MB")
        print(f"    - Final Hdr  : {os.path.basename(audio_h_out)}")
        print(f"    - Est. Ratio : {a_ratio:.1f} %")
    else:
        print("   ERROR: Files missing. Check logs above.")
    print("="*50)

if __name__ == "__main__":
    main()
