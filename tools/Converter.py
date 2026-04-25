# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import shutil
import csv
import argparse
from datetime import datetime

def run_cmd(cmd):
    print(f"\n[Running]: {cmd}")
    return subprocess.run(cmd, shell=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--param", help="Path to config CSV file to load settings")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    os.chdir(root_dir)
    py3 = f'"{sys.executable}"'

    # パラメータ初期化
    params = {
        "url": "",
        "width": 128, "height": 64, "label": "128x64",
        "mode": "d",
        "gauss_sigma": 1.0, "edge_repair": 3.0,
        "contrast": 1.5, 
        "brightness": 0.0,  # ブライトネス変数を初期化
        "unsharp": 1.5,
        "audio_gain_db": 0.0, "dynamic_boost_enabled": "y", "sample_rate": 18000,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # --- パラメータロード ---
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

    if not is_auto:
        print("===============================================")
        print("   ESP32-C3 Video & Audio Header Builder       ")
        print("===============================================")
        params["url"] = input("\n[1] YouTube URL: ").strip()
        if not params["url"]: return

        sizes = [("64x32", 64, 32), ("72x40", 72, 40), ("128x64", 128, 64), ("256x128", 256, 128), ("230x240", 230, 240)]
        for i, (l, _, _) in enumerate(sizes, 1): print(f"  {i}: {l}")
        idx = int(input("Choice [3]: ") or "3") - 1
        params["label"], params["width"], params["height"] = sizes[idx]
        
        m_input = input("\n[2] Rendering Mode (d: dither / g: gaussian) [d]: ").lower() or "d"
        params["mode"] = "g" if m_input.startswith('g') else "d"
        if params["mode"] == "g":
            params["gauss_sigma"] = float(input("    - Gaussian Sigma [1.0]: ") or "1.0")
            params["edge_repair"] = float(input("    - Edge Repair [3.0]: ") or "3.0")
        
        # --- ビデオ補正入力 (ブライトネスを追加) ---
        params["contrast"] = float(input("\n[3] Enhancement - Contrast [1.5]: ") or "1.5")
        params["brightness"] = float(input("    Enhancement - Brightness (-1.0 to 1.0) [0.0]: ") or "0.0")
        params["unsharp"] = float(input("    Enhancement - Basic Unsharp [1.5]: ") or "1.5")
        
        params["audio_gain_db"] = float(input("\n[4] Audio - Gain (dB) [0.0]: ") or "0.0")
        params["sample_rate"] = round(int(input("    Sample Rate [18000]: ") or "18000") / 500) * 500

    # パス設定
    res_dir = os.path.normpath(os.path.join(root_dir, "resouce"))
    inc_dir = os.path.normpath(os.path.join(root_dir, "include"))
    frames_dir = os.path.join(res_dir, "Frames", params["label"])
    temp_gray_dir = os.path.join(res_dir, "temp_gray")
    os.makedirs(frames_dir, exist_ok=True); os.makedirs(temp_gray_dir, exist_ok=True); os.makedirs(inc_dir, exist_ok=True)

    video_mp4 = os.path.join(res_dir, "download.mp4")
    temp_bad, temp_wav = os.path.join(res_dir, "output.bad"), os.path.join(res_dir, "audio.wav")
    codec_py, wave_py = os.path.join(script_dir, "Codec.py"), os.path.join(script_dir, "Wave2AdpcmH.py")
    video_h_out, audio_h_out = os.path.join(inc_dir, "bad_data.h"), os.path.join(inc_dir, "adpcm4.h")

    # --- 処理実行 ---
    run_cmd(f'yt-dlp -f "bestvideo[height<=480]+bestaudio/best" --merge-output-format mp4 -o "{video_mp4}" {params["url"]}')

    # Video Filter (brightnessをeqフィルタに反映)
    f_list = [f"scale={params['width']}:{params['height']}", "format=gray"]
    if params["mode"] == "g":
        f_list.append(f"unsharp=5:5:{params['edge_repair']}:5:5:0")
        f_list.append(f"gblur=sigma={params['gauss_sigma']}")
    else:
        f_list.append(f"unsharp=5:5:{params['unsharp']}:5:5:0")
    
    # ここで contrast と brightness を適用
    f_list.append(f"eq=contrast={params['contrast']}:brightness={params['brightness']}")
    f_list.append("lutrgb='r=if(lt(val,32),0,if(gt(val,223),255,val))'")
    
    run_cmd(f'ffmpeg -y -i "{video_mp4}" -vf "{",".join(f_list)}" -vcodec bmp "{temp_gray_dir}/g_%04d.bmp"')

    final_vf = "format=monob" if params["mode"] == "d" else "threshold=128,format=monob"
    run_cmd(f'ffmpeg -y -i "{temp_gray_dir}/g_%04d.bmp" -vf "{final_vf}" -vcodec bmp "{frames_dir}/frame_%04d.bmp"')
    shutil.rmtree(temp_gray_dir)

    # --- 後続処理 (エンコード・音声・統計) ---
    bmp_list = [f for f in os.listdir(frames_dir) if f.endswith('.bmp')]
    v_raw_size = sum(os.path.getsize(os.path.join(frames_dir, f)) for f in bmp_list)
    run_cmd(f'{py3} "{codec_py}" -t e -p "{frames_dir}/" -n frame_ -s 0001 -e {len(bmp_list):04d} -o "{temp_bad}"')
    v_comp_size = os.path.getsize(temp_bad)

    a_filters = ["loudnorm=I=-16:TP=-1.5:LRA=11"]
    if params["dynamic_boost_enabled"] == 'y': a_filters.insert(0, "compand=attacks=0:points=-80/-80|-20/-5|-10/-1|0/0")
    if params["audio_gain_db"] != 0: a_filters.insert(0, f"volume={params['audio_gain_db']}dB")
    
    run_cmd(f'ffmpeg -y -i "{video_mp4}" -af "{",".join(a_filters)}" -vn -ar {params["sample_rate"]} -ac 1 -f wav "{temp_wav}"')
    a_raw_size = os.path.getsize(temp_wav)
    a_comp_size = (a_raw_size - 44) // 4
    run_cmd(f'{py3} "{wave_py}" -i "{temp_wav}" -o "{audio_h_out}" -r {params["sample_rate"]}')
    run_cmd(f'{py3} "{codec_py}" -t c -i "{temp_bad}" -H "{video_h_out}"')

    # 最終統計とCSV保存
    v_ratio = (v_comp_size / v_raw_size) * 100 if v_raw_size > 0 else 0
    stats = {
        "枚数": len(bmp_list),
        "動画raw容量": f"{v_raw_size} B",
        "動画圧縮後": f"{v_comp_size} B",
        "動画圧縮率": f"{v_raw_size}/{v_comp_size} ({v_ratio:.2f}%)",
        "音声圧縮容量": f"{a_comp_size} B",
        "合計バイナリサイズ": f"{v_comp_size + a_comp_size} B"
    }
    params.update(stats)
    cfg_fn = f"param_{datetime.now().strftime('%Y%m%d_%H%M%S')}.cfg"
    with open(os.path.join(script_dir, cfg_fn), 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f); [writer.writerow([k, v]) for k, v in params.items()]

    print("\n" + "="*60)
    print(f"   FINISH: {cfg_fn} | Brightness: {params['brightness']}")
    print(f"   Total Size: {v_comp_size + a_comp_size} B")
    print("="*60)

if __name__ == "__main__":
    main()
