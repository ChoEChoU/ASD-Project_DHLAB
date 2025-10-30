import os
import glob
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm

# 🔧 경로 인자 설정
parser = argparse.ArgumentParser()
parser.add_argument(
    '--root_dir',
    type=str,
    default='./data/preprocessed/unmatching_features',
    help='Path to pkl files (features source)'
)
parser.add_argument(
    '--target_dir',
    type=str,
    default='./data/preprocessed/unmatching_npy',
    help='Where to save merged npy files'
)
args = parser.parse_args()

root_dir = args.root_dir.rstrip('/')   # 예: ./data/preprocessed/unmatching_features
target_dir = args.target_dir.rstrip('/')  # 예: ./data/preprocessed/unmatching_npy

# 모든 instance 폴더 경로 수집 (e.g., .../P001/video_001/instance_0.pkl)
vid_dirs = sorted({os.path.dirname(p) for p in glob.glob(f'{root_dir}/*/*/instance_*.pkl')})

for vid in tqdm(vid_dirs, desc="Merging PKL → NPY"):
    ins_list = sorted(glob.glob(f'{vid}/instance_*.pkl'))

    features = []
    for ins in ins_list:
        with open(ins, 'rb') as f:
            ins_feat = pickle.load(f)
            features.append(ins_feat)

    if not features:
        continue

    # torch.Tensor 리스트 → [N, ...] tensor
    features = torch.stack(features)

    # 상대 경로 기준 출력 파일 생성
    relative_path = os.path.relpath(vid, root_dir)
    output_path = os.path.join(target_dir, f"{relative_path}.npy")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not os.path.exists(output_path):
        np.save(output_path, features)
    else:
        print(f"⏭️  Skipping existing file: {output_path}")

print("✅ 모든 PKL → NPY 변환 완료!")