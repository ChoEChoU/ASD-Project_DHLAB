from pathlib import Path
import shutil

# ✅ 설정
target_task = "I"   # 예: 'A' Task
target_month = "18"  # 예: '08' 개월 영상만 필터링

input_root = Path("./data/1017_unmatching_npy")
output_root = Path(f"./data/1017_splits/task_{target_task}/cell_month_{target_month}/input_npy")
output_root.mkdir(parents=True, exist_ok=True)

# ✅ 필터링 및 복사
matched_paths = []

for path in input_root.rglob("*.npy"):
    filename = path.stem  # 예: 'AF08-1710369515137'
    prefix = filename.split("-")[0]  # 예: 'AF08'
    
    task_match = prefix[0]              # 'A'
    month_match = ''.join(filter(str.isdigit, prefix))[-2:]  # '08'

    if task_match == target_task and month_match == target_month:
        # 상위 구조 유지하여 output 경로 설정
        relative_path = path.relative_to(input_root)
        destination = output_root / relative_path

        # 상위 폴더 생성
        destination.parent.mkdir(parents=True, exist_ok=True)

        # 파일 복사
        shutil.copy2(path, destination)
        matched_paths.append(str(destination))

# ✅ 결과 출력
print(f"✅ {len(matched_paths)}개 파일이 Task {target_task}, 개월 수 {target_month}에 해당하며 복사되었습니다.")