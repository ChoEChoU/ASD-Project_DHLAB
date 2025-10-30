#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================
# 🔧 기본 설정
# ============================================
BASE_DIR="./data/splits"
TRUE_LABEL="./data/lists/normal_patients.csv"
SCRIPT="./data_preprocess/8_create_true_label_csv.py"

# ============================================
# 🔧 Task-월 매핑
# ============================================
declare -A TASK_MONTH_MAP
TASK_MONTH_MAP["AF"]="02 04 06 08 10 14 16 18"
TASK_MONTH_MAP["D"]="04 06 08 10 12 14 16 18"
TASK_MONTH_MAP["G"]="10 12 14 16 18"
TASK_MONTH_MAP["H"]="10 12 14 16"
TASK_MONTH_MAP["I"]="12 14 16 18"

# ============================================
# 🔁 모든 Task / Month 조합 실행
# ============================================
for TASK in "${!TASK_MONTH_MAP[@]}"; do
    for M in ${TASK_MONTH_MAP[$TASK]}; do
        VIDEO_ROOT="${BASE_DIR}/task_${TASK}/cell_month_${M}/input_npy"
        OUTPUT_CSV="${BASE_DIR}/task_${TASK}/cell_month_${M}/input_label.csv"

        if [ -d "$VIDEO_ROOT" ]; then
            echo "▶️ 실행: Task=$TASK, Month=$M"
            python3 "$SCRIPT" \
                --video_root "$VIDEO_ROOT" \
                --true_label_csv "$TRUE_LABEL" \
                --output_csv "$OUTPUT_CSV"
        else
            echo "⏭️ 스킵 (폴더 없음): $VIDEO_ROOT"
        fi
    done
done

echo "🎉 모든 Task / Month 라벨 CSV 생성 완료!"