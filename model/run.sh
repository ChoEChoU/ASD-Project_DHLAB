#!/bin/bash
set -euo pipefail

# 실행 위치는 프로젝트 루트(ASD-Project_DHLAB)라고 가정
cd "$(dirname "$0")"
cd ..

# RFE_VALUES=(20 15 10 0)
RFE_VALUES=(0)
# GRID_OPTIONS=("" "--grid")
GRID_OPTIONS=("")
# METRIC_OPTIONS=("f1" "auc")
METRIC_OPTIONS=("auc")
# IMPUTE_OPTIONS=("zero" "growing")
IMPUTE_OPTIONS=("growing")

DATA_DIR="./data/final_data"
SAVE_ROOT="./model_results_matching"
GROUP_NAME="Matched_E_Cell"
MODALITY="multimodal/video_demo"
SPLITS=5

for rfe in "${RFE_VALUES[@]}"; do
  for grid in "${GRID_OPTIONS[@]}"; do
    for metric in "${METRIC_OPTIONS[@]}"; do
      for impute in "${IMPUTE_OPTIONS[@]}"; do
        echo "▶ 실행: RFE=${rfe}, GRID=${grid:-no}, METRIC=${metric}, IMPUTE=${impute}"
        python3 model/main-matching_Grid.py \
          --rfe "${rfe}" \
          ${grid} \
          --metric "${metric}" \
          --impute "${impute}" \
          --data_dir "${DATA_DIR}" \
          --save_root "${SAVE_ROOT}" \
          --group_name "${GROUP_NAME}" \
          --modality "${MODALITY}" \
          --n_splits "${SPLITS}"
      done
    done
  done
done

echo "🎉 Grid 러닝 완료"