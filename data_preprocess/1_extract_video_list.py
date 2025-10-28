import pandas as pd
from pathlib import Path

# 🔧 설정
video_roots = [
    Path("./data/2025_06_updated/aiai_baby_upload/videos"),
    Path("./data/2025_06_updated/aiai_baby_upload/2025-05-31/videos"),
    Path("./data/2025_06_updated/aiai_baby_upload/2025-06-30/videos"),
    Path("./data/2025_06_updated/aiai_baby_upload/2025-07-31/videos"),
    Path("./data/2025_06_updated/aiai_baby_upload/2025-08-31/videos"),
]  # ✅ 여러 경로 지정

normal_csv_path = "./data/1017_splits/normal_patients.csv"
others_csv_path = "./data/1017_splits/other_patients.csv"
output_csv_path = "./data/1017_splits/unmatching_video_list.csv"

# ✅ 1. 정상군 / 기타 환자 ID 로드
normal_df = pd.read_csv(normal_csv_path)
others_df = pd.read_csv(others_csv_path)

normal_ids = set(normal_df['patient_id'].astype(str))
others_ids = set(others_df['patient_id'].astype(str))
all_patient_ids = normal_ids.union(others_ids)

# ✅ 2. 여러 video_root 순회
filtered_records = []

for video_root in video_roots:
    video_paths = list(video_root.glob("*/*.mp4"))
    print(f"📂 {video_root} 내 비디오 {len(video_paths)}개 확인")

    for path in video_paths:
        patient_id = path.parent.name
        video_filename = path.stem  # 확장자 제외

        if patient_id not in all_patient_ids:
            continue

        filtered_records.append({
            "video_path": str(path),
            "patient_id": patient_id,
            "label": 0 if patient_id in normal_ids else 1  # 0 = Normal, 1 = Others
        })

# ✅ 3. 저장
df_out = pd.DataFrame(filtered_records)
df_out.to_csv(output_csv_path, index=False)

print(f"\n✅ 총 {len(df_out)}개 비디오 추출 및 저장 완료 → {output_csv_path}")