# data_preprocess/1_extract_video_list.py
import pandas as pd
from pathlib import Path
import re
import os

# 🔧 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # ASD-Project_DHLAB/
DATA_ROOT = PROJECT_ROOT / "data"

video_root = DATA_ROOT / "raws" / "videos"
normal_csv_path = DATA_ROOT / "lists" / "normal_patients.csv"
others_csv_path = DATA_ROOT / "lists" / "other_patients.csv"
output_csv_path = DATA_ROOT / "lists" / "unmatching_video_list.csv"

# ✅ 1. 정상군 환자 ID 로드
normal_df = pd.read_csv(normal_csv_path)
others_df = pd.read_csv(others_csv_path)

normal_ids = set(normal_df['patient_id'].astype(str))
others_ids = set(others_df['patient_id'].astype(str))

all_patient_ids = normal_ids.union(others_ids)

# ✅ 2. 비디오 경로 순회
video_paths = list(video_root.glob("*/*.mp4"))

filtered_records = []

for path in video_paths:
    current_video_path = str(path)
    patient_id = path.parent.name
    video_filename = path.stem  # 확장자 제외

    if patient_id not in all_patient_ids:
        continue

    if video_filename.startswith(""):
        filtered_records.append({
            "video_path": current_video_path,
            "patient_id": patient_id,
            "label": 0 if patient_id in normal_ids else 1  # 0 = Normal, 1 = Others
        })

# ✅ 3. 저장
df_out = pd.DataFrame(filtered_records)
df_out.to_csv(output_csv_path, index=False)

print(f"✅ Saved filtered video list → {output_csv_path}")
print(f"Total videos: {len(df_out)}")