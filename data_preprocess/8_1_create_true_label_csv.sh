#!/bin/bash

MONTHS=("12" "14" "16" "18")
BASE="./data/1017_splits/task_AF"
TRUE_LABEL="./data/1017_splits/normal_patients.csv"
SCRIPT="./data/1017_splits/utils/8_create_true_label_csv.py"

for M in "${MONTHS[@]}"; do
    VIDEO_ROOT="$BASE/cell_month_${M}/input_npy"
    OUTPUT_CSV="$BASE/cell_month_${M}/input_label.csv"

    if [ -d "$VIDEO_ROOT" ]; then
        echo "실행: month=$M"
        python "$SCRIPT" \
            --video_root "$VIDEO_ROOT" \
            --true_label_csv "$TRUE_LABEL" \
            --output_csv "$OUTPUT_CSV"
    else
        echo "스킵 (폴더 없음): $VIDEO_ROOT"
    fi
done