import os
import pandas as pd

# 경로 설정
csv_folder_path = "./outputs/1017_no_match_results"  # 여러 CSV 파일이 들어있는 폴더 경로
normal_list_csv = '../data/1017_splits/normal_patients.csv'  # normal군 patient_id 리스트가 들어있는 CSV

# normal list 불러오기
normal_df = pd.read_csv(normal_list_csv)
normal_patients = set(normal_df['patient_id'].astype(str))

# 폴더 내 모든 CSV 파일 순회
for filename in os.listdir(csv_folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(csv_folder_path, filename)
        df = pd.read_csv(file_path)

        # patient_id가 normal list에 있으면 group=1, 아니면 group=0
        df['patient_id'] = df['patient_id'].astype(str)
        df['group'] = df['patient_id'].apply(lambda pid: 0 if pid in normal_patients else 1)

        # 수정된 파일 저장 (원본 덮어쓰기 or 다른 폴더에 저장 가능)
        df.to_csv(file_path, index=False)
        print(f"Updated: {filename}")