import os
import glob
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, required=True, help='Path to pkl files (videos_features)')
parser.add_argument('--target_dir', type=str, required=True, help='Where to save merged npy files')
args = parser.parse_args()

root_dir = args.root_dir.rstrip('/')  # 예: ./data/1017_unmatching_features
target_dir = args.target_dir.rstrip('/')  # 예: ./data/1017_unmatching_npy

# 모든 instance 폴더 (e.g., .../H110-*/instance_0.pkl → 상위폴더 추출)
vid_dirs = sorted({os.path.dirname(p) for p in glob.glob(f'{root_dir}/*/*/instance_*.pkl')})

for vid in tqdm(vid_dirs):
    ins_list = sorted(glob.glob(f'{vid}/instance_*.pkl'))

    feature = []
    for ins in ins_list:
        with open(ins, 'rb') as f:
            ins_feat = pickle.load(f)
            feature.append(ins_feat)

    if not feature:
        continue

    feature = torch.stack(feature)

    # 상대 경로 생성
    relative_path = os.path.relpath(vid, root_dir)
    output_path = os.path.join(target_dir, f"{relative_path}.npy")

    # 폴더 없으면 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not os.path.exists(output_path):
        np.save(output_path, feature)
    else:
        print(f"⏭️  Skipping existing file: {output_path}")