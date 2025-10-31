# data_preprocess/14_sum_results_csv.py
import os
import argparse
from glob import glob
import pandas as pd

SPLITS = ["test"]

def parse_exp_folder(folder_name: str):
    """
    folder_name 예시:
      - task_AF_month_02_groupE_match_fold0
      - task_D_month_18_groupE_no_match_fold3
    반환: dict(task='AF', month='02', mode='match'|'no_match', fold=0)
    """
    parts = folder_name.split("_")
    info = {"task": None, "month": None, "mode": None, "fold": None}
    for i, p in enumerate(parts):
        if p == "task" and i + 1 < len(parts):
            info["task"] = parts[i + 1]
        if p == "month" and i + 1 < len(parts):
            info["month"] = parts[i + 1]
        if p == "groupE" and i + 1 < len(parts):
            info["mode"] = parts[i + 1]  # match or no_match
        if p.startswith("fold"):
            try:
                info["fold"] = int(p.replace("fold", ""))
            except ValueError:
                pass
    return info

def collect_fold_results(results_base_dir: str, fold: int):
    """
    fold에 해당하는 모든 실험 폴더를 찾고, split별 result csv 파일 경로 목록을 반환
    반환: {split: [(prefix, path), ...], ...}
      - prefix: "AF_02" 같은 (task_month) 접두어
    """
    out = {s: [] for s in SPLITS}
    pattern = os.path.join(results_base_dir, f"task_*_month_*_groupE_*_fold{fold}")
    exp_dirs = sorted(glob(pattern))

    for exp_dir in exp_dirs:
        folder = os.path.basename(exp_dir)
        meta = parse_exp_folder(folder)
        task, month = meta["task"], meta["month"]
        if task is None or month is None:
            print(f"⚠️ 폴더명 파싱 실패: {folder}")
            continue
        prefix = f"{task}_{month}"

        # split 접두어로 파일 찾기
        for split in SPLITS:
            # train_result.csv / valid_result.csv / test_result.csv 형태 기대
            cand = glob(os.path.join(exp_dir, f"{split}*_result.csv"))
            for path in cand:
                out[split].append((prefix, path))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_base_dir", type=str, default="./outputs_csv",
                    help="결과 CSV를 모아둔 루트(폴더 구조 보존해 복사한 경우 outputs_csv 권장)")
    ap.add_argument("--folds_base_dir", type=str, default="./data/1017_splits/no_matching_fold",
                    help="fold별 train/valid/test 환자 CSV가 있는 루트")
    ap.add_argument("--output_dir", type=str, default="./outputs_csv_summaries",
                    help="병합 결과 저장 경로")
    ap.add_argument("--num_folds", type=int, default=5)
    ap.add_argument("--prob_col", type=str, default="prob_class_0",
                    help="병합 대상 확률 컬럼명 (기본: prob_class_0)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for fold in range(args.num_folds):
        print(f"\n📁 Processing fold_{fold}")

        # 1) fold의 split별 환자 리스트 준비
        subject_ids_by_split = {}
        for split in SPLITS:
            fold_csv = os.path.join(args.folds_base_dir, f"fold_{fold}", f"{split}.csv")
            df = pd.read_csv(fold_csv)
            subject_ids_by_split[split] = pd.DataFrame({"patient_id": df["patient_id"].astype(str)})

        # 2) 병합 초기화 (split별)
        merged_by_split = {s: subject_ids_by_split[s].copy() for s in SPLITS}

        # 3) fold 결과 수집
        results_by_split = collect_fold_results(args.results_base_dir, fold)
        # 디버그: 발견한 파일 경로 목록
        for s in SPLITS:
            if results_by_split[s]:
                print(f"  - {s}: {len(results_by_split[s])} files")

        # 4) split별로 결과 병합
        for split in SPLITS:
            for prefix, result_path in results_by_split[split]:
                df = pd.read_csv(result_path)
                if "patient_id" not in df.columns:
                    print(f"⚠️ patient_id 없음: {result_path}")
                    continue
                if args.prob_col not in df.columns:
                    print(f"⚠️ 필요한 열 없음({args.prob_col}) → 건너뜀: {result_path}")
                    continue

                sub = df[["patient_id", args.prob_col]].copy()
                sub["patient_id"] = sub["patient_id"].astype(str)
                sub.rename(columns={args.prob_col: f"{prefix}_{args.prob_col}"}, inplace=True)

                merged_by_split[split] = pd.merge(
                    merged_by_split[split], sub, on="patient_id", how="left"
                )

        # 5) 저장
        for split in SPLITS:
            save_path = os.path.join(args.output_dir, f"{split}_fold{fold}.csv")
            merged_by_split[split].to_csv(save_path, index=False)
            print(f"✅ 저장 완료: {save_path}")

if __name__ == "__main__":
    main()