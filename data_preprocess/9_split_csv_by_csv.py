import os
import json
import pandas as pd
from pathlib import Path

# ==== 경로/설정 ====
BASE_SPLIT_DIR = Path("./data/splits")                 # 7단계 출력 베이스
FOLDS_ROOT = Path("./data/folds/matching_fold")     # fold_0, fold_1, ... 안에 train/valid/test.csv
LABEL_MAP = {0: "Normal", 1: "Others"}

# Task → (출력 task폴더명, 허용 month)
TASK_MONTH_MAP = {
    "A": ("AF", ["02", "04", "06", "08", "10"]),
    "F": ("AF", ["14", "16", "18"]),
    "D": ("D",  ["04", "06", "08", "10", "12", "14", "16", "18"]),
    "G": ("G",  ["10", "12", "14", "16", "18"]),
    "H": ("H",  ["10", "12", "14", "16"]),
    "I": ("I",  ["12", "14", "16", "18"]),
}

def build_for_one(task_key: str, out_task: str, month: str):
    """
    task_key = 원본 파일명 접두(예: A/F/D/G/H/I)
    out_task = 출력 폴더명(AF/D/G/H/I)
    month = "02" 등 2자리 스트링
    """
    # 입력 CSV (Step 8 결과)
    all_csv_path = BASE_SPLIT_DIR / f"task_{out_task}" / f"cell_month_{month}" / "input_label.csv"
    if not all_csv_path.exists():
        print(f"⏭️ 스킵(입력 CSV 없음): {all_csv_path}")
        return

    # 출력 디렉토리
    output_dir = BASE_SPLIT_DIR / f"task_{out_task}" / f"cell_month_{month}" / "input_data_matching"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 전체 데이터 로딩
    df_all = pd.read_csv(all_csv_path)
    # VideoPath = "{patient_id}/{video_id}" 형식이라고 가정
    df_all["VideoID"] = df_all["VideoPath"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    df_all["patient_id"] = df_all["VideoPath"].apply(lambda x: x.split("/")[0])
    df_all["Label"] = df_all["Label"].astype(int)

    stats_records = []
    all_test_patients = []

    # 각 fold 순회
    for fold_name in sorted(os.listdir(FOLDS_ROOT)):
        fold_path = FOLDS_ROOT / fold_name
        if not fold_path.is_dir():
            continue
        try:
            fold_idx = int(str(fold_name).split("_")[-1])
        except Exception:
            print(f"⚠️ fold 폴더명 해석 실패, 스킵: {fold_name}")
            continue

        print(f"\n🔁 Fold {fold_idx}")

        # fold의 train/valid/test CSV 읽기
        df_split = {}
        for split in ["train", "valid", "test"]:
            csv_path = fold_path / f"{split}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"필요 CSV 없음: {csv_path}")
            df_split[split] = pd.read_csv(csv_path)
            df_split[split]["Split"] = split

        # test 환자 집합 추적(중복 검증용)
        test_subjects = set(df_split["test"]["patient_id"])
        all_test_patients.append(test_subjects)

        # split별 환자 ID를 기준으로 df_all에서 필터링
        parts = []
        for split in ["train", "valid", "test"]:
            subjects = df_split[split]["patient_id"]
            sub = df_all[df_all["patient_id"].isin(subjects)].copy()
            sub["Split"] = split
            parts.append(sub)

        df_all_split = pd.concat(parts, ignore_index=True)

        # JSON 생성
        database = {}
        for _, row in df_all_split.iterrows():
            database[row["VideoID"]] = {
                "subset": row["Split"],
                "annotations": [{"segment": ["0", "0"], "label": LABEL_MAP[row["Label"]]}],
            }
        json_out = output_dir / f"gt_fold{fold_idx}.json"
        with open(json_out, "w") as f:
            json.dump({"database": database}, f, indent=4)

        # split txt 저장
        for split in ["train", "valid", "test"]:
            txt_path = output_dir / f"split_{split}_fold_{fold_idx}.txt"
            video_ids = df_all_split[df_all_split["Split"] == split]["VideoID"].tolist()
            with open(txt_path, "w") as f:
                f.write("\n".join(video_ids))

        # 통계 기록/출력
        for split in ["train", "valid", "test"]:
            part = df_all_split[df_all_split["Split"] == split]
            sample_total = len(part)
            sample_normal = (part["Label"] == 0).sum()
            sample_others = (part["Label"] == 1).sum()

            patient_group = part.groupby("patient_id")["Label"].first()
            patient_total = len(patient_group)
            patient_normal = (patient_group == 0).sum()
            patient_others = (patient_group == 1).sum()

            stats_records.append({
                "task": out_task,
                "month": month,
                "fold": fold_idx,
                "split": split,
                "sample_total": sample_total,
                "sample_normal": sample_normal,
                "sample_others": sample_others,
                "patient_total": patient_total,
                "patient_normal": patient_normal,
                "patient_others": patient_others,
            })

            # 콘솔 요약
            def pct(a, b): return (a / b) if b else 0.0
            print(f"📊 [Fold {fold_idx} - {split.upper()}] "
                  f"Samples {sample_total} | N:{sample_normal}({pct(sample_normal, sample_total):.1%}) "
                  f"| O:{sample_others}({pct(sample_others, sample_total):.1%})  "
                  f"Patients {patient_total} | N:{patient_normal}({pct(patient_normal, patient_total):.1%}) "
                  f"| O:{patient_others}({pct(patient_others, patient_total):.1%})")

    # 통계 CSV 저장 (task/month 단위)
    stats_df = pd.DataFrame(stats_records)
    stats_df.to_csv(output_dir / "fold_split_statistics.csv", index=False)

    # Test fold 간 환자 중복 확인
    test_union = set()
    for i, test_set in enumerate(all_test_patients):
        overlap = test_union & test_set
        # if overlap:
            # raise ValueError(f"❌ Fold {i}의 test set과 이전 fold와 중복됨: {overlap}")
        test_union.update(test_set)

    print(f"✅ 완료: task={out_task}, month={month} → {output_dir}")

def main():
    # 모든 Task/Month 순회
    for task_key, (out_task, months) in TASK_MONTH_MAP.items():
        for m in months:
            build_for_one(task_key, out_task, m)

if __name__ == "__main__":
    main()