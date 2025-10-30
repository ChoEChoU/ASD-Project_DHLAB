#!/usr/bin/env bash
set -Eeuo pipefail

# =====================================
# 🔧 공통 설정
# =====================================
BASE="./data/splits"

# Task별 Month 매핑 (Step 7과 동일)
declare -A TASK_MONTH_MAP
TASK_MONTH_MAP["AF"]="02 04 06 08 10 14 16 18"
TASK_MONTH_MAP["D"]="04 06 08 10 12 14 16 18"
TASK_MONTH_MAP["G"]="10 12 14 16 18"
TASK_MONTH_MAP["H"]="10 12 14 16"
TASK_MONTH_MAP["I"]="12 14 16 18"

# =====================================
# 🔁 모든 Task × Month 반복
# =====================================
for TASK in "${!TASK_MONTH_MAP[@]}"; do
    for M in ${TASK_MONTH_MAP[$TASK]}; do
        SRC="${BASE}/task_${TASK}/cell_month_${M}/input_npy"           # 원본 NPY
        DEST="${BASE}/task_${TASK}/cell_month_${M}/input_data_matching/features/rgb"

        if [[ -d "$SRC" ]]; then
            mkdir -p "$DEST"
            echo "▶ 복사 실행: task=${TASK}, month=${M}"
            cp -r "${SRC}"/*/* "$DEST" 2>/dev/null || echo "⚠️ 복사할 파일 없음: ${SRC}"
        else
            echo "⏭️ 스킵 (폴더 없음): ${SRC}"
        fi
    done
done

echo "🎉 모든 Task × Month NPY feature 복사 완료!"