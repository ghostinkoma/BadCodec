# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import shutil
import csv
import argparse
import json
from datetime import datetime

def run_cmd(cmd):
    print(f"\n[Running]: {cmd}")
    return subprocess.run(cmd, shell=True)

def get_video_fps(video_path):
    """ffprobeを使用して動画の本来のフレームレートを取得する"""
    if not os.path.exists(video_path):
        return 25.0
    cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of json "{video_path}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    try:
        data = json.loads(result.stdout)
        rate = data['streams'][0]['r_frame_rate']
        if '/' in rate:
            num, den = map(int, rate.split('/'))
            return round(num / den, 2)
        return float(rate)
    except:
        return 25.0  # 取得失敗時のフォールバック

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--param", help="Path to config CSV file to load settings")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    os.chdir(root_dir)
    py3 = f'"{sys.executable}"'

    # 1. パスとディレクトリの設定
    res_dir = os.path.normpath(os.path.join(root_dir, "resouce"))
    inc_dir = os.path.normpath(os.path.join(root_dir, "include"))
    frames_parent = os.path.join(res_dir, "Frames")
    temp_gray_dir = os.path.join(res_dir, "temp_gray")
    
    video_mp4 = os.path.join(res_dir, "download.mp4")
    temp_bad = os.path.join(res_dir, "output.bad")
    temp_wav = os.path.join(res_dir, "audio.wav")
    
    codec_py = os.path.join(script_dir, "Codec.py")
    wave_py = os.path.join(script_dir, "Wave2AdpcmH.py")
    video_h_out = os.path.join(inc_dir, "bad_data.h")
    audio_h_out = os.path.join(inc_dir, "adpcm4.h")

    # 【重要】バグ修正：以前の生成物・中間ファイルを完全にクリア
    print(">>> Cleaning up previous work files...")
    for f in [video_mp4, temp_bad, temp_wav]:
        if os.path.exists(f):
            os.remove(f)
    for d in [frames_parent, temp_gray_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)

    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(inc_dir, exist_ok=True)

    # 2. パラメータ初期化
    params = {
        "url": "",
        "width": 128, "height": 64, "label": "128x64",
        "original_fps": 0.0,
        "fps": 0.0,
        "mode": "d",
        "threshold_pct": 50,     
        "gauss_sigma": 1.0, 
        "edge_repair": 3.0,
        "contrast": 1.5, 
        "brightness": 0.0, 
        "unsharp": 1.5,
        "audio_gain_db": 0.0, 
        "dynamic_boost_enabled": "y", 
        "sample_rate": 18000,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # パラメータロード
    is_auto = False
    if args.param and os.path.exists(args.param):
        print(f">>> Loading settings from {args.param}...")
        with open(args.param, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            loaded_data = {row[0]: row[1] for row in reader}
            for k, v in params.items():
                if k in loaded_data:
                    if isinstance(v, float): params[k] = float(loaded_data[k])
                    elif isinstance(v, int): params[k] = int(loaded_data[k])
                    else: params[k] = loaded_data[k]
            if params["url"]: is_auto = True

    # 3. VIDEO DOWNLOAD
    if not is_auto:
        print("===============================================")
        print("   ESP32-C3 Video & Audio Header Builder       ")
        print("===============================================")
        params["url"] = input("\n[1] YouTube URL: ").strip()
        if not params["url"]: return

    # --force-overwrites を追加して、古いファイルによる中断を防止
    run_cmd(f'yt-dlp --force-overwrites -f "bestvideo[height<=480]+bestaudio/best" --merge-output-format mp4 -o "{video_mp4}" {params["url"]}')
    
    if not os.path.exists(video_mp4):
        print("Error: Video download failed.")
        return

    # 本来のFPSを取得
    params["original_fps"] = get_video_fps(video_mp4)

    # 4. ユーザー入力 (非オートモード時)
    if not is_auto:
        sizes = [("64x32", 64, 32), ("72x40", 72, 40), ("128x64", 128, 64), ("256x128", 256, 128), ("230x240", 230, 240)]
        for i, (l, _, _) in enumerate(sizes, 1): print(f"  {i}: {l}")
        idx = int(input("Choice [3]: ") or "3") - 1
        params["label"], params["width"], params["height"] = sizes[idx]
        
        print(f"\n[1.5] Frame Rate Setting:")
        print(f"    - Original FPS: {params['original_fps']}")
        params["fps"] = float(input(f"    - Change FPS to [{params['original_fps']}]: ") or params["original_fps"])
        
        m_input = input("\n[2] Mode (d: dither / g: gaussian / b: binary) [d]: ").lower() or "d"
        params["mode"] = m_input[0] if m_input[0] in ['d', 'g', 'b'] else 'd'
        
        if params["mode"] == "g":
            params["gauss_sigma"] = float(input("    - Gaussian Sigma (Dot Distribution) [1.0]: ") or "1.0")
            params["edge_repair"] = float(input("    - Edge Repair (Unsharp) [3.0]: ") or "3.0")
        elif params["mode"] == "b":
            params["threshold_pct"] = int(input("    - Binary Threshold (0-100%) [50]: ") or "50")
        
        params["contrast"] = float(input("\n[3] Enhancement - Contrast [1.5]: ") or "1.5")
        params["brightness"] = float(input("    Enhancement - Brightness [0.0]: ") or "0.0")
        params["unsharp"] = float(input("    Enhancement - Basic Unsharp [1.5]: ") or "1.5")
        
        params["audio_gain_db"] = float(input("\n[4] Audio - Gain (dB) [0.0]: ") or "0.0")
        params["sample_rate"] = round(int(input("    Sample Rate [18000]: ") or "18000") / 500) * 500

    # 5. フォルダ再構築
    frames_dir = os.path.join(frames_parent, params["label"])
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(temp_gray_dir, exist_ok=True)

    # 6. VIDEO PROCESSING (高精度FPS変換 + フィルタ)
    f_list = [f"fps={params['fps']}", f"scale={params['width']}:{params['height']}", "format=gray"]
    f_list.append(f"eq=contrast={params['contrast']}:brightness={params['brightness']}")
    f_list.append(f"unsharp=5:5:{params['unsharp']}:5:5:0")
    
    if params["mode"] == "g":
        f_list.append(f"gblur=sigma={params['gauss_sigma']}")
        f_list.append(f"unsharp=5:5:{params['edge_repair']}:5:5:0")
    
    # グレースケール中間生成
    run_cmd(f'ffmpeg -y -i "{video_mp4}" -vf "{",".join(f_list)}" -vcodec bmp "{temp_gray_dir}/g_%04d.bmp"')

    # 二値化 / ディザ処理
    final_vf = "format=monob"
    if params["mode"] == "b":
        thresh_val = int(params["threshold_pct"] * 2.55)
        final_vf = f"lutrgb='r=if(lt(val,{thresh_val}),0,255):g=if(lt(val,{thresh_val}),0,255):b=if(lt(val,{thresh_val}),0,255)',format=monob"
    
    run_cmd(f'ffmpeg -y -i "{temp_gray_dir}/g_%04d.bmp" -vf "{final_vf}" -vcodec bmp "{frames_dir}/frame_%04d.bmp"')
    
    # 中間ディレクトリ削除
    if os.path.exists(temp_gray_dir):
        shutil.rmtree(temp_gray_dir)

    # 7. VIDEO & AUDIO CONVERT (Codec / Wave2Adpcm)
    bmp_list = sorted([f for f in os.listdir(frames_dir) if f.endswith('.bmp')])
    if not bmp_list:
        print("Error: No frames generated.")
        return

    v_raw_size = sum(os.path.getsize(os.path.join(frames_dir, f)) for f in bmp_list)
    
    # Codec.py を叩いて .bad バイナリを生成
    run_cmd(f'{py3} "{codec_py}" -t e -p "{frames_dir}/" -n frame_ -s 0001 -e {len(bmp_list):04d} -o "{temp_bad}"')
    
    if not os.path.exists(temp_bad):
        print("Error: Codec.py failed to create .bad file.")
        return
    
    v_comp_size = os.path.getsize(temp_bad)

    # 音声フィルタ設定
    a_filters = ["loudnorm=I=-16:TP=-1.5:LRA=11"]
    if params["dynamic_boost_enabled"] == 'y':
        a_filters.insert(0, "compand=attacks=0:points=-80/-80|-20/-5|-10/-1|0/0")
    if params["audio_gain_db"] != 0:
        a_filters.insert(0, f"volume={params['audio_gain_db']}dB")
    
    # WAV出力
    run_cmd(f'ffmpeg -y -i "{video_mp4}" -af "{",".join(a_filters)}" -vn -ar {params["sample_rate"]} -ac 1 -f wav "{temp_wav}"')
    
    if not os.path.exists(temp_wav):
        print("Error: FFmpeg failed to create WAV file.")
        return

    a_raw_size = os.path.getsize(temp_wav)
    a_comp_size = (a_raw_size - 44) // 4  # ADPCMの期待サイズ計算
    
    # ヘッダーファイル(.h)生成
    run_cmd(f'{py3} "{wave_py}" -i "{temp_wav}" -o "{audio_h_out}" -r {params["sample_rate"]}')
    run_cmd(f'{py3} "{codec_py}" -t c -i "{temp_bad}" -H "{video_h_out}"')

    # 8. レポートと設定保存
    v_ratio = (v_comp_size / v_raw_size) * 100 if v_raw_size > 0 else 0
    total_binary = v_comp_size + a_comp_size
    params.update({
        "枚数": len(bmp_list), "動画raw容量": f"{v_raw_size} B", "動画圧縮後": f"{v_comp_size} B",
        "動画圧縮率": f"{v_ratio:.2f}%", "音声圧縮容量": f"{a_comp_size} B",
        "合計バイナリサイズ": f"{total_binary} B"
    })

    cfg_fn = f"param_{datetime.now().strftime('%Y%m%d_%H%M%S')}.cfg"
    with open(os.path.join(script_dir, cfg_fn), 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for k, v in params.items():
            writer.writerow([k, v])

    print("\n" + "="*60)
    print(f"   ENCODE FINISHED: {cfg_fn}")
    print("-" * 60)
    print(f"   Original FPS: {params['original_fps']}")
    print(f"   Converted FPS: {params['fps']}")
    print(f"   Resolution: {params['label']}")
    print(f"   Mode: {params['mode'].upper()}")
    print(f"   Total Binary Size: {total_binary/1024/1024:.2f} MB")
    print("="*60)

if __name__ == "__main__":
    main()
