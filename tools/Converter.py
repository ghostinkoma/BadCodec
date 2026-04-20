import os
import subprocess
import sys

def run_cmd(cmd):
    print(f"\n[Running]: {cmd}")
    # shell=TrueはWindowsの環境変数パスを通すために使用
    return subprocess.run(cmd, shell=True)

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    clear_screen()
    print("===============================================")
    print("   ESP32-C3 BadApple Video Converter Pro       ")
    print("===============================================")

    # 1. YouTube URL 入力
    url = input("\n[1] YouTube URL: ").strip()
    if not url:
        print("URLを入力してください。")
        return

    # 2. 画像サイズ選択
    print("\n[2] Select Image Size:")
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

    # 3. 音声サンプルレート設定 (8000 - 32000, 500 step)
    print("\n[3] Audio Sample Rate (8000Hz - 32000Hz, 500Hz step):")
    print("  Recommended: 22050")
    raw_rate = int(input("Input Hz [22050]: ") or "22050")
    # 500Hz単位に丸める
    sample_rate = round(raw_rate / 500) * 500
    sample_rate = max(8000, min(32000, sample_rate))

    # 4. 作業ディレクトリ
    work_dir = "./resouce"
    print(f"\n[4] Working Directory: {work_dir}")

    # 最終確認
    print("\n" + "-"*30)
    print(f" Target URL: {url}")
    print(f" Size      : {label}")
    print(f" Audio     : {sample_rate} Hz")
    print(f" Dir       : {work_dir}")
    print("-"*30)

    confirm = input("\nDo you want to convert with these params? (y/n): ").lower()
    if confirm != 'y':
        print("Canceled.")
        return

    # --- 実行セクション ---
    frames_dir = f"{work_dir}/Frames/{label}"
    os.makedirs(frames_dir, exist_ok=True)
    video_path = f"{work_dir}/download.mp4"
    bad_file = f"{work_dir}/output.bad"
    wav_path = f"{work_dir}/audio.wav"

    # ダウンロード
    print("\n>>> Downloading...")
    run_cmd(f'yt-dlp -f "bestvideo[height<=480]+bestaudio/best" --merge-output-format mp4 -o "{video_path}" {url}')

    # フレーム抽出 & 2値化
    print("\n>>> Extracting Frames...")
    run_cmd(f'ffmpeg -y -i "{video_path}" -vf "scale={width}:{height},format=gray,geq=\'lum=if(gt(lum(X,Y),128),255,0)\'" -vcodec bmp "{frames_dir}/frame_%04d.bmp"')

    # フレーム数取得
    frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.bmp')]
    frame_count = len(frame_files)

    # 映像エンコード
    print("\n>>> Encoding Video...")
    run_cmd(f'python Codec.py -t e -p "{frames_dir}/" -n frame_ -s 0001 -e {frame_count:04d} -o "{bad_file}"')

    # 音声処理
    print("\n>>> Processing Audio...")
    run_cmd(f'ffmpeg -y -i "{video_path}" -vn -ar {sample_rate} -ac 1 -f wav "{wav_path}"')
    run_cmd(f'python Wave2AdpcmH.py "{wav_path}" ./include/adpcm4.h')

    # ヘッダー化
    print("\n>>> Finalizing Header...")
    run_cmd(f'python Codec.py -t c -i "{bad_file}" -H ./include/bad_data.h')

    print("\n" + "="*40)
    print("  ALL TASKS COMPLETED!")
    print(f"  Total Frames: {frame_count}")
    print(f"  Flash usage will be around: {os.path.getsize(bad_file)//1024} KB")
    print("="*40)

if __name__ == "__main__":
    main()
