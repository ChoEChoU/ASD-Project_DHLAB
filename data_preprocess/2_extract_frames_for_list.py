import cv2
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
# 설정
CSV_PATH = "./data/1017_splits/unmatching_video_list_2.csv"
OUTPUT_ROOT = Path("./data/1017_unmatching_videos_2")
INSTANCE_LEN = 32  # 한 인스턴스당 프레임 수
NUM_WORKERS = 2
# 비디오 → 인스턴스별 프레임 저장 함수
def process_video(row):
    video_path = Path(row['video_path'])
    patient_id = row['patient_id']
    video_id = video_path.stem
    output_dir = OUTPUT_ROOT / patient_id / video_id

    if (output_dir / "instance_0").exists():
        return f"⏭️ 건너뜀: {video_id} (이미 처리됨)"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return f"❌ 열기 실패: {video_path}"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    usable_frames = total_frames - (total_frames % INSTANCE_LEN)
    if usable_frames < INSTANCE_LEN:
        cap.release()
        return f"⚠️ 프레임 부족: {video_id} ({total_frames}프레임)"

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
    return f"✅ 완료: {video_id} ({num_instances} instances)"
# 메인
def main():
    df = pd.read_csv(CSV_PATH)
    with Pool(processes=NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap_unordered(process_video, df.to_dict(orient="records")),
                            total=len(df),
                            desc="비디오 처리 중"))
    for r in results:
        print(r)
if __name__ == "__main__":
    main()