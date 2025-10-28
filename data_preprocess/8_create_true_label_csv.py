import pandas as pd
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--video_root', required=True, help='Root path to video folders')
parser.add_argument('--true_label_csv', required=True, help='Path to CSV with normal patient IDs')
parser.add_argument('--output_csv', required=True, help='Where to save output CSV with labels')
args = parser.parse_args()

# ✅ 정상 환자 ID 리스트 로드
true_df = pd.read_csv(args.true_label_csv)
normal_patient_ids = set(true_df['patient_id'].astype(str).tolist())

# ✅ 비디오 폴더 순회
video_root = Path(args.video_root)
video_paths = sorted(video_root.glob("*/*"))  # e.g., B-01-xxxx/H110-xxxx

records = []

for vid_path in video_paths:
    patient_id = Path(vid_path.name).stem
    parent_folder = vid_path.parent.name  # 상위 폴더명: B-01-230615-18

    label = 0 if parent_folder in normal_patient_ids else 1
    video_id = f"{parent_folder}/{patient_id}"
    print(video_id)

    records.append({'VideoPath': video_id, 'Label': label})

# ✅ CSV 저장
output_df = pd.DataFrame(records)
output_df.to_csv(args.output_csv, index=False)

print(f"✅ 라벨링 CSV 저장 완료: {args.output_csv}")