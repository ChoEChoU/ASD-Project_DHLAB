import os
import shutil

def copy_csv_files_with_structure(src_dir, dst_dir):
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.csv'):
                src_file_path = os.path.join(root, file)
                # 상대 경로 계산
                relative_path = os.path.relpath(src_file_path, src_dir)
                dst_file_path = os.path.join(dst_dir, relative_path)
                dst_folder = os.path.dirname(dst_file_path)

                # 대상 폴더가 없으면 생성
                os.makedirs(dst_folder, exist_ok=True)
                # 파일 복사
                shutil.copy2(src_file_path, dst_file_path)
                print(f"Copied: {src_file_path} -> {dst_file_path}")

# 사용 예시
src_folder = './outputs/1017_no_match'
dst_folder = './outputs/1017_no_match_csv'
os.makedirs(dst_folder, exist_ok=True)

copy_csv_files_with_structure(src_folder, dst_folder)