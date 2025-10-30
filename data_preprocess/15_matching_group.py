# data_preprocess/15_matching_group.py
import os
import argparse
import pandas as pd

def add_group_column(csv_folder_path: str, normal_list_csv: str, overwrite: bool = True):
    """CSV 폴더 내 파일에 normal/others 그룹 구분 컬럼 추가"""
    # 정상군 ID 리스트 로드
    if not os.path.exists(normal_list_csv):
        raise FileNotFoundError(f"❌ normal list CSV not found: {normal_list_csv}")
    normal_df = pd.read_csv(normal_list_csv)
    normal_patients = set(normal_df["patient_id"].astype(str))
    print(f"✅ Loaded {len(normal_patients)} normal patient IDs")

    # CSV 폴더 내 파일 처리
    csv_files = [f for f in os.listdir(csv_folder_path) if f.endswith(".csv")]
    if not csv_files:
        print(f"⚠️ No CSV files found in {csv_folder_path}")
        return

    for filename in csv_files:
        file_path = os.path.join(csv_folder_path, filename)
        df = pd.read_csv(file_path)

        if "patient_id" not in df.columns:
            print(f"⚠️ Skipping {filename}: 'patient_id' column missing")
            continue

        df["patient_id"] = df["patient_id"].astype(str)
        df["group"] = df["patient_id"].apply(lambda pid: 0 if pid in normal_patients else 1)

        if overwrite:
            save_path = file_path
        else:
            base, ext = os.path.splitext(filename)
            save_path = os.path.join(csv_folder_path, f"{base}_with_group{ext}")

        df.to_csv(save_path, index=False)
        print(f"✅ Updated: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Add group column (0=Normal, 1=Others) to result CSVs.")
    parser.add_argument("--csv_folder", type=str, default="./outputs_csv_summaries",
                        help="Folder containing fold result CSV files")
    parser.add_argument("--normal_list", type=str, default="./data/1017_splits/normal_patients.csv",
                        help="Path to normal patients CSV")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing files (default: False → create *_with_group.csv)")
    args = parser.parse_args()

    add_group_column(args.csv_folder, args.normal_list, args.overwrite)

if __name__ == "__main__":
    main()