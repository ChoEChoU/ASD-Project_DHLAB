import os
import pandas as pd
from glob import glob

# === 사용자 경로 설정 ===
results_base_dir = "./outputs/1017_no_match_csv"
folds_base_dir = "../data/1017_splits/no_matching_fold"
output_dir = "./outputs/1017_no_match_results"
os.makedirs(output_dir, exist_ok=True)

splits = ["train", "valid", "test"]
num_folds = 5

# === fold별로 반복 ===
for fold in range(num_folds):
    print(f"\n📁 Processing fold_{fold}")
    
    # 1. fold의 split별 환자 리스트 (patient_id → patient_id)
    subject_ids_by_split = {}
    for split in splits:
        fold_csv = os.path.join(folds_base_dir, f"fold_{fold}", f"{split}.csv")
        df = pd.read_csv(fold_csv)
        subject_ids_by_split[split] = pd.DataFrame({
            "patient_id": df["patient_id"].astype(str)
        })

    # 2. split별 병합 결과 초기화
    merged_by_split = {split: subject_ids_by_split[split].copy() for split in splits}

    # 3. 해당 fold 결과 파일 찾기
    result_paths = glob(
        f"{results_base_dir}/task_*_groupE_no_match_fold{fold}/*_result.csv", recursive=True
    )

    print(result_paths)

    for result_path in result_paths:
        filename = os.path.basename(result_path)
        folder = os.path.basename(os.path.dirname(result_path))

        # split 추출
        split = next((s for s in splits if filename.startswith(s)), None)
        if not split:
            continue

        # task/month 추출
        try:
            parts = folder.split('_')
            task = parts[1]   # A
            month = parts[3]  # 02
            prefix = f"{task}_{month}"
        except IndexError:
            print(f"⚠️ 폴더명 파싱 실패: {folder}")
            continue

        # CSV 읽고 필요한 열만 추출
        df = pd.read_csv(result_path)
        if "patient_id" not in df.columns:
            print(f"⚠️ patient_id 없음: {result_path}")
            continue

        required_cols = ["patient_id", "prob_class_0"]#, "prob_class_1"]
        if not all(col in df.columns for col in required_cols):
            print(f"⚠️ 필요한 열 없음 → 건너뜀: {result_path}")
            continue

        df = df[required_cols]
        df["patient_id"] = df["patient_id"].astype(str)

        # 열 이름에 prefix 붙이기
        df = df.rename(columns={
            "prob_class_0": f"{prefix}_prob_class_0",
            # "prob_class_1": f"{prefix}_prob_class_1"
        })

        # 병합
        merged_by_split[split] = pd.merge(
            merged_by_split[split], df, on="patient_id", how="left"
        )

    # 4. 결과 저장
    for split in splits:
        save_path = os.path.join(output_dir, f"{split}_fold{fold}.csv")
        merged_by_split[split].to_csv(save_path, index=False)
        print(f"✅ 저장 완료: {save_path}")