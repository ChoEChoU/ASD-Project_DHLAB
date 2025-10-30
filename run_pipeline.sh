# #!/usr/bin/env bash
# # ============================================================
# # 🧠 AIAI End-to-End Pipeline (Steps 1~18)
# # ------------------------------------------------------------
# # 전체 개요:
# #  - Steps 1~10: 비디오 데이터 전처리 및 feature 생성
# #  - Steps 11~15: CASE 모델 학습 및 결과 정리
# #  - Steps 16~18: 인구통계 설문 병합 + 최종 ML 학습
# # ------------------------------------------------------------
# # 실행 위치 : 프로젝트 루트 (예: /ASD-Project_DHLAB)
# # 로그 파일 : logs/pipeline_YYYYMMDD_HHMMSS.log
# # ============================================================

# set -Eeuo pipefail

# # -----------------------------
# # ✅ 0) 환경 설정
# # -----------------------------
# cd "$(dirname "$0")"
# ROOT="$(pwd)"
# LOG_DIR="$ROOT/logs"
# mkdir -p "$LOG_DIR"
# LOG_FILE="$LOG_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log"
# exec > >(tee -a "$LOG_FILE") 2>&1

# echo "📍 WORKDIR: $ROOT"
# echo "🗒️ LOG    : $LOG_FILE"

# step() {
#   echo ""
#   echo "========================================"
#   echo "[$(date '+%F %T')] STEP $1 :: $2"
#   echo "========================================"
# }

# # ============================================================
# # 🧩 [1~10단계] 비디오 → 프레임 → Feature 추출 파이프라인
# # ============================================================

# # 1) 모든 비디오 경로를 스캔하여 리스트 CSV 생성
# #    출력: data/lists/unmatching_video_list.csv
# step 1 "Extract video list"
# python3 data_preprocess/1_extract_video_list.py

# # 2) 각 비디오(.mp4)를 프레임 단위로 분해 (1초 간격)
# #    출력: data/preprocessed/unmatching_videos_frames/
# step 2 "Extract frames for each video"
# python3 data_preprocess/2_extract_frames_for_list.py

# # 3) 프레임 디렉토리를 기반으로 instance list (.txt) 작성
# #    각 영상의 경로 및 label 정보를 포함
# #    출력: data/lists/unmatched_instance_list.txt
# step 3 "Build unmatched instance list"
# python3 data_preprocess/3_get_instance_list.py

# # 4) SlowFast (mmaction2)로 영상 feature(.pkl) 추출
# #    CONFIG / CKPT는 SlowFast pretrained 설정 사용
# #    출력: 각 프레임 폴더 하위에 .pkl 파일 생성
# CONFIG="model/mmaction2/configs/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb.py"
# CKPT="model/mmaction2/ckpts/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb_20220818-1cb6dfc8.pth"
# FEAT_OUT="data/preprocessed/unmatching_videos_frames"
# step 4 "Extract SlowFast clip features"
# python3 model/mmaction2/tools/misc/clip_feature_extraction.py "$CONFIG" "$CKPT" "$FEAT_OUT"

# # 5) 프레임 폴더에서 생성된 .pkl 파일만 구조 유지하며 이동
# #    입력: data/preprocessed/unmatching_videos_frames
# #    출력: data/preprocessed/unmatching_features
# step 5 "Move .pkl features to structured dir"
# bash data_preprocess/5_move_pkl.sh

# # 6) 각 instance별 .pkl feature를 병합하여 하나의 .npy로 저장
# #    입력: data/preprocessed/unmatching_features
# #    출력: data/preprocessed/unmatching_npy
# step 6 "Merge .pkl → .npy files"
# python3 data_preprocess/6_create_npy_files.py \
#   --root_dir ./data/preprocessed/unmatching_features \
#   --target_dir ./data/preprocessed/unmatching_npy

# # 7) Task / Month 기준으로 NPY를 분류하여 복사
# #    예: task_AF, task_D, task_G, ...
# #    출력: data/splits/task_*/cell_month_*/input_npy/
# step 7 "Split NPY files by Task & Month"
# python3 data_preprocess/7_select_month_task_npy.py

# # 8) 각 Task/Month별로 정상군 라벨 CSV 생성
# #    입력: normal_patients.csv
# #    출력: input_label.csv (0=Normal, 1=Others)
# step 8 "Create label CSVs for each Task/Month"
# bash data_preprocess/8_1_create_true_label_csv.sh

# # 9) Fold별 CSV를 기반으로 split txt/json/statistics 생성
# #    출력: gt_foldX.json, split_train/valid/test_fold_X.txt
# step 9 "Generate split txt/json for 5-fold structure"
# python3 data_preprocess/9_split_csv_by_csv.py

# # 10) 학습용 feature 구조로 정리 (복사)
# #     출력: input_data_no_matching/features/rgb/
# step 10 "Copy feature npy files to training structure"
# bash data_preprocess/10_mv_features_for_train.sh

# echo ""
# echo "✅ Preprocessing completed (Steps 1~10)."

# # ============================================================
# # 🧩 [11~15단계] CASE 모델 학습 및 결과 요약
# # ============================================================

# # 11) CASE 모델 학습 (matching dataset)
# step 11 "CASE model training"
# bash model/CASE/11_run_2025_aiai_AF_group_match_E_cell_all_matching.sh

# # 12) CASE 모델 inference-only 실행 (테스트셋)
# step 12 "CASE inference_only"
# bash model/CASE/12_run_2025_aiai_AF_group_match_E_cell_all_test_matching.sh

# 13) outputs/ 아래에서 CSV 파일만 추출 (폴더 구조 유지)
#     출력: outputs_csv/
step 13 "Collect only result CSVs from outputs"
python3 model/CASE/13_copy_only_results_csv.py \
  --src ./outputs \
  --dst ./outputs_csv

# 14) fold별로 train/valid/test split 기준으로 결과 병합
#     입력: outputs_csv + data/folds/matching_fold
#     출력: outputs_csv_summaries/
step 14 "Summarize results by fold/split"
python3 data_preprocess/14_sum_results_csv.py \
  --results_base_dir ./outputs_csv \
  --folds_base_dir ./data/folds/matching_fold \
  --output_dir ./outputs_csv_summaries \
  --num_folds 5 \
  --prob_col prob_class_0

# 15) 정상군 리스트(normal_patients.csv)를 이용해
#     group 컬럼 추가 (0=Normal, 1=Others)
step 15 "Add group column (Normal/Others)"
python3 data_preprocess/15_matching_group.py \
  --csv_folder ./outputs_csv_summaries \
  --normal_list ./data/lists/normal_patients.csv \
  --overwrite

echo "🎉 CASE results aggregation completed (Steps 11~15)."

# ============================================================
# 🧩 [16~18단계] 설문 병합 및 최종 ML 학습
# ============================================================

# 16) Demography 설문 전처리 (P1~P5)
#     입력: ./data/Demography_all.xlsx
#     출력: ./data/preprocessed/tabular/Demo_processed.csv
step 16 "Preprocess Demography survey (P1~P5)"
python3 data_preprocess/16_preprocessing_demo.py

# 17) 설문 + 모델 요약 결과 병합 (LEFT JOIN on patient_id)
#     입력: outputs_csv_summaries + Demo_processed.csv
#     출력: data/final_data/
step 17 "Merge Demography with model results"
python3 data_preprocess/17_merge_demo_with_results.py \
  --results_dir ./outputs_csv_summaries \
  --demo_csv ./data/preprocessed/tabular/Demo_processed.csv \
  --output_dir ./data/final_data \
  --key patient_id

# 18) 최종 머신러닝 모델 학습 (GridSearch/RFE/F1/AUC 조합)
#     실행: model/run.sh
#     출력: ./model_results_matching/
step 18 "Run final matching ML model (RFE/Grid/Metric/Impute sweep)"
bash model/run.sh

echo ""
echo "🎯 ✅ Pipeline fully completed (Steps 1–18)"
echo "📦 Final merged data : ./data/final_data/"
echo "📊 Model results     : ./model_results_matching/"
echo "🧩 Logs              : $LOG_FILE"