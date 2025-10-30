# data_preprocess/17_merge_demo_with_results.py
import os
import argparse
import pandas as pd
from glob import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo_csv",
        type=str,
        default="./data/preprocessed/tabular/Demo_processed.csv",
        help="인구통계 전처리 CSV 경로"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./outputs_csv_summaries",
        help="fold별 통합 결과 CSV들이 들어있는 폴더"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/final_data",
        help="최종 병합 CSV 저장 폴더"
    )
    parser.add_argument(
        "--key",
        type=str,
        default="patient_id",
        help="merge 기준 컬럼명"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------------------
    # 1️⃣ Demo 데이터 로드
    # ---------------------------
    print(f"📥 Loading demo data: {args.demo_csv}")
    demo_df = pd.read_csv(args.demo_csv)
    demo_df[args.key] = demo_df[args.key].astype(str)

    # ---------------------------
    # 2️⃣ Result CSV들 병합
    # ---------------------------
    result_csvs = sorted(glob(os.path.join(args.results_dir, "*.csv")))
    print(f"🔍 Found {len(result_csvs)} result CSVs in {args.results_dir}")

    for csv_path in result_csvs:
        name = os.path.basename(csv_path)
        print(f"➡️ Merging with: {name}")

        result_df = pd.read_csv(csv_path)
        result_df[args.key] = result_df[args.key].astype(str)

        merged = pd.merge(result_df, demo_df, on=args.key, how="left")

        save_path = os.path.join(args.output_dir, name)
        merged.to_csv(save_path, index=False)
        print(f"✅ Saved: {save_path}")

    print("\n🎉 All merged CSVs saved in:", args.output_dir)

if __name__ == "__main__":
    main()