import pandas as pd
from pathlib import Path

# 입력 디렉토리
input_dir = Path("./data/1017_unmatching_videos")

# 정상군 CSV
normal_csv_path = "./data/1017_splits/normal_patients.csv"
normal_df = pd.read_csv(normal_csv_path)
normal_patient_ids = set(normal_df['patient_id'].astype(str))

# 출력 경로
output_txt = "./data/1017_splits/matched_instance_list.txt"
output_lines = []

def extract_instance_index(p: Path):
    try:
        return int(p.name.split("_")[1])
    except:
        return float('inf')

# 모든 patient_id 폴더 순회
for patient_folder in sorted(input_dir.iterdir()):
    if not patient_folder.is_dir():
        continue

    patient_id = patient_folder.name
    label = 0 if patient_id in normal_patient_ids else 1  # ✅ 라벨 결정

    # 각 영상 폴더 순회 (e.g., H110-... 등)
    for video_folder in sorted(patient_folder.iterdir()):
        if not video_folder.is_dir():
            continue

        # instance 폴더 숫자 기준 정렬
        instance_folders = sorted(video_folder.glob("instance_*"), key=extract_instance_index)

        for instance_folder in instance_folders:
            if not instance_folder.is_dir():
                continue
            output_lines.append(f"{instance_folder} {label}")

# 저장
with open(output_txt, "w") as f:
    for line in output_lines:
        f.write(line + "\n")

print(f"✅ 총 {len(output_lines)}개 인스턴스를 숫자 순으로 저장했습니다: {output_txt}")