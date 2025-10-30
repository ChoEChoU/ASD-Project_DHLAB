import cv2
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import os

# 🔧 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"

# 입력/출력 경로
CSV_PATH = DATA_ROOT / "lists" / "unmatching_video_list.csv"          # Step 1 결과
VIDEO_ROOT = DATA_ROOT / "raws" / "videos"                            # 실제 비디오 경로
OUTPUT_ROOT = DATA_ROOT / "preprocessed" / "unmatching_videos_frames" # 프레임 저장 경로

INSTANCE_LEN = 32   # 한 인스턴스당 프레임 수
NUM_WORKERS = 2

def process_video(row):
    """
    각 비디오(mp4)를 32프레임 단위로 instance_x 폴더에 나눠 저장
    """
    patient_id = str(row["patient_id"])
    video_name = Path(row["video_path"]).stem  # 원본 CSV에 저장된 파일명
    video_path = VIDEO_ROOT / patient_id / f"{video_name}.mp4"
    output_dir = OUTPUT_ROOT / patient_id / video_name

    if not video_path.exists():
        return f"❌ 파일 없음: {video_path}"

    # 이미 처리된 경우 건너뛰기
    if (output_dir / "instance_0").exists():
        return f"⏭️ 건너뜀: {video_name} (이미 처리됨)"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return f"❌ 열기 실패: {video_path}"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    usable_frames = total_frames - (total_frames % INSTANCE_LEN)
    if usable_frames < INSTANCE_LEN:
        cap.release()
        return f"⚠️ 프레임 부족: {video_name} ({total_frames}프레임)"

    num_instances = usable_frames // INSTANCE_LEN

    for i in range(num_instances):
        instance_path = output_dir / f"instance_{i}"
        instance_path.mkdir(parents=True, exist_ok=True)

        for j in range(INSTANCE_LEN):
            ret, frame = cap.read()
            if not ret:
                break
            filename = instance_path / f"img_{j+1:05d}.jpg"
            cv2.imwrite(str(filename), frame)

    cap.release()
    return f"✅ 완료: {video_name} ({num_instances} instances)"

def main():
    df = pd.read_csv(CSV_PATH)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    with Pool(processes=NUM_WORKERS) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(process_video, df.to_dict(orient="records")),
                total=len(df),
                desc="비디오 처리 중"
            )
        )

    for r in results:
        print(r)

if __name__ == "__main__":
    main()