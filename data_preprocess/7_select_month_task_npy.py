from pathlib import Path
import shutil
from tqdm import tqdm

# 🔧 입력 루트
input_root = Path("./data/preprocessed/unmatching_npy")

# 🔧 Task-월 매핑
task_month_map = {
    "A": {"output_task": "AF", "months": ["02", "04", "06", "08", "10"]},
    "F": {"output_task": "AF", "months": ["14", "16", "18"]},
    "D": {"output_task": "D",  "months": ["04", "06", "08", "10", "12", "14", "16", "18"]},
    "G": {"output_task": "G",  "months": ["10", "12", "14", "16", "18"]},
    "H": {"output_task": "H",  "months": ["10", "12", "14", "16"]},
    "I": {"output_task": "I",  "months": ["12", "14", "16", "18"]},
}

# 🔧 출력 루트 베이스
split_root = Path("./data/splits")

# ----------------------------------------------------------

def copy_filtered_files(task: str, month: str, output_task: str):
    """특정 Task와 Month에 해당하는 .npy만 필터링하여 복사"""
    output_root = split_root / f"task_{output_task}" / f"cell_month_{month}" / "input_npy"
    output_root.mkdir(parents=True, exist_ok=True)

    matched_paths = []
    for path in input_root.rglob("*.npy"):
        filename = path.stem  # 예: AF08-1710369515137
        prefix = filename.split("-")[0]
        task_match = prefix[0]
        month_match = ''.join(filter(str.isdigit, prefix))[-2:]

        if task_match == task and month_match == month:
            relative_path = path.relative_to(input_root)
            destination = output_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            matched_paths.append(str(destination))

    print(f"✅ [{task}] {month}개월 → {len(matched_paths)}개 파일 복사 완료 → {output_root}")

# ----------------------------------------------------------

def main():
    total_count = 0
    for task, info in task_month_map.items():
        output_task = info["output_task"]
        for month in info["months"]:
            copy_filtered_files(task, month, output_task)
            total_count += 1
    print(f"\n🎉 전체 Task/Month 조합 {total_count}개 처리 완료!")

# ----------------------------------------------------------

if __name__ == "__main__":
    main()