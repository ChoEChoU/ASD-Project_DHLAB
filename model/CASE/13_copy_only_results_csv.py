# model/CASE/13_copy_only_results.py
import os
import shutil
import argparse

def copy_csv_files_with_structure(src_dir: str, dst_dir: str):
    copied = 0
    for root, _, files in os.walk(src_dir):
        for file in files:
            # ✅ result 파일만 선택 (필요시 확장 가능)
            if not file.lower().endswith(".csv"):
                continue
            src_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(src_file_path, src_dir)
            dst_file_path = os.path.join(dst_dir, relative_path)
            os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)
            shutil.copy2(src_file_path, dst_file_path)
            print(f"📄 Copied: {src_file_path} -> {dst_file_path}")
            copied += 1
    print(f"\n✅ 완료: 총 {copied}개의 CSV 파일 복사됨.")

def main():
    parser = argparse.ArgumentParser(description="Copy only .csv files from outputs preserving folder structure.")
    parser.add_argument("--src", default="./outputs", help="Source directory (default: ./outputs)")
    parser.add_argument("--dst", default="./outputs_csv", help="Destination directory (default: ./outputs_csv)")
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    copy_csv_files_with_structure(args.src, args.dst)

if __name__ == "__main__":
    main()