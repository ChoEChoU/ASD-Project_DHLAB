#!/usr/bin/env bash
# model/CASE/12_run_2025_aiai_AF_group_match_E_cell_all_test_matching.sh
set -Eeuo pipefail

# GPU (외부에서 CUDA_VISIBLE_DEVICES=0 bash ... 로 덮어쓰기 가능)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# 스크립트 위치 기준 프로젝트 루트로 이동
cd "$(dirname "$0")/../.."

PY="python3"
ENTRY="model/CASE/main_case.py"

# 공통 하이퍼파라미터
MODAL="rgb"
NUM_SEG=30
NUM_CLUST=5
EPOCHS=300
DET_STEP=20
BATCH=1
FOLDS=(0 1 2 3 4)

# Task ↔ Month 매핑
declare -A TASK_MONTH_MAP
TASK_MONTH_MAP["AF"]="02 04 06 08 10 14 16 18"
TASK_MONTH_MAP["D"]="04 06 08 10 12 14 16 18"
TASK_MONTH_MAP["G"]="10 12 14 16 18"
TASK_MONTH_MAP["H"]="10 12 14 16"
TASK_MONTH_MAP["I"]="12 14 16 18"

for TASK in AF D G H I; do
  for M in ${TASK_MONTH_MAP[$TASK]}; do
    BASE="./data/splits/task_${TASK}/cell_month_${M}"
    DATA_PATH="${BASE}/input_data_matching"
    LABEL_CSV="${BASE}/input_label.csv"

    if [[ ! -d "$DATA_PATH" || ! -f "$LABEL_CSV" ]]; then
      echo "⏭️  스킵: ${DATA_PATH} 또는 ${LABEL_CSV} 없음"
      continue
    fi

    for F in "${FOLDS[@]}"; do
      EXP="task_${TASK}_month_${M}_groupE_match_fold${F}"
      echo "▶ Inference: ${EXP}"
      $PY "$ENTRY" \
        --exp_name "$EXP" \
        --data_path "$DATA_PATH/" \
        --patient_csv "$LABEL_CSV" \
        --modal "$MODAL" \
        --num_segments "$NUM_SEG" \
        --num_clusters "$NUM_CLUST" \
        --num_epochs "$EPOCHS" \
        --detection_inf_step "$DET_STEP" \
        --fold "$F" \
        --inference_only \
        --batch_size "$BATCH"
    done
  done
done

echo "🎉 Inference for all Task × Month × Fold 완료"