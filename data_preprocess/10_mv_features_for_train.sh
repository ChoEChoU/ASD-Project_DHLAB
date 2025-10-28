#!/bin/bash
set -euo pipefail

MONTHS=("12" "14" "16" "18")
TASK="task_I"
BASE="./data/1017_splits/${TASK}"

for M in "${MONTHS[@]}"; do
    SRC="${BASE}/cell_month_${M}/input_npy/*/*"
    DEST="${BASE}/cell_month_${M}/input_data_no_matching/features/rgb"

    if [[ -d "${BASE}/cell_month_${M}" ]]; then
        mkdir -p "$DEST"
        echo "▶ 복사 실행: month=${M}"
        cp $SRC "$DEST"
    else
        echo "스킵: ${BASE}/cell_month_${M} 없음"
    fi
done