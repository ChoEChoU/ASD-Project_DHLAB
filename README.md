# ASD-Project_DHLAB

영상 기반 행동 피처와 인구통계 설문 데이터를 통합하여  
자폐 스펙트럼 장애(ASD) 예측 모델을 구축하는 **End-to-End 파이프라인**입니다.  

이 프로젝트는 **행동 비디오(raw)** 와 **설문(Demography)** 데이터를 입력으로 받아  
프레임 추출 → SlowFast 기반 feature 추출 → CASE 모델 inference →  
인구통계 데이터 병합 → 최종 ML 기반 ASD 예측까지 수행합니다.

---

## Installation

### 1. Clone this repository
```bash
git clone https://github.com/ChoEChoU/ASD-Project_DHLAB.git
cd ASD-Project_DHLAB
```

### 2. Run the full pipeline (자동 환경 설치 포함)
`run_pipeline.sh` 스크립트는 실행 시 다음을 자동으로 처리합니다:

- Conda 환경(`asd_env`, Python 3.8.20) 생성 및 활성화  
- CUDA 11.3 기반 PyTorch 설치 (`torch==1.12.1+cu113`)  
- OpenMMLab 핵심 라이브러리(`mmcv==2.0.0`) 설치  
- Requirements.txt 의존성 설치  
- MMACTION2 editable 설치 (`model/mmaction2`)  
- 이후 1~18단계 전체 파이프라인 실행

```bash
bash run_pipeline.sh
```

환경 세팅부터 결과 예측까지 자동으로 수행됨

---

## Directory Structure

```plaintext
ASD-Project_DHLAB
├── data
│   ├── raws/
│   │   ├── videos/
│   │   │   └── patient_id/
│   │   │       ├── A010-1234.mp4
│   │   │       ├── D010-1124.mp4
│   │   │       ├── ...
│   │   │       └── G010-1235.mp4
│   │   └── demo/
│   │       └── Demography_all.xlsx
│   ├── preprocessed/
│   │   ├── unmatching_videos_frames/
│   │   ├── unmatching_features/
│   │   ├── unmatching_npy/
│   │   └── tabular/
│   │       └── Demo_processed.csv
│   ├── splits/
│   │   └── task_*/cell_month_*/input_npy/
│   ├── lists/
│   │   ├── normal_patients.csv 
│   │   └── other_patients.csv
│   ├── folds/
│   │   └── matching_fold/fold_*/
│   └── final_data/
│       └── test_fold{i}.csv
│
├── model
│   ├── mmaction2/                # SlowFast 기반 Feature extractor
│   ├── CASE/                     # CASE 모델 학습 및 inference
│   ├── ml_weight/                # ML 학습된 모델 Weights
│   ├── main-matching_Grid.py     # ML 학습/튜닝 (GridSearch, RFE 등)
│   ├── 18_inference.py           # 최종 ML inference
│   └── run.sh                    # ML 실험 자동 실행
│
├── outputs/                      # CASE 원본 결과 저장
│
├── outputs_csv/                  # outputs/ 에서 CSV만 추출된 폴더
├── outputs_csv_summaries/        # fold별 결과 요약 저장
│
├── logs/
│   └── pipeline_*.log
│
├── data_preprocess/              # 데이터 전처리 코드
│
├── Prediction_Results/           # 최종 예측 결과
│
└── run_pipeline.sh               # 전체 파이프라인 실행 스크립트
```

---

## Video Naming Convention

- 파일명은 **4글자 prefix** 로 시작해야 함.  
  - 앞 1~2글자: **Task명** (예: A, D, F, G, H, I …)  
  - 뒤 2글자: **개월 수** (예: 02, 04, 06, 08, 10 …)  
- 예시:
  - `A010-1234.mp4` → Task A, 10개월
  - `D008-5678.mp4` → Task D, 8개월  

---

## Pipeline Overview

| Step | Description |
|------|--------------|
| **1** | 비디오 리스트 추출 (`data/lists/unmatching_video_list.csv`) |
| **2** | 각 비디오를 프레임 단위로 분리 (`unmatching_videos_frames/`) |
| **3** | 프레임 폴더 기반 instance list 작성 (`.txt`) |
| **4** | SlowFast (mmaction2)로 Feature(.pkl) 추출 |
| **5** | `.pkl` 파일만 구조 유지하여 이동 |
| **6** | Instance-level `.pkl` → `.npy` 병합 저장 |
| **7** | Task / Month 기준으로 데이터 분류 |
| **8** | 각 Month별 라벨링 CSV 생성 |
| **9~10** | (Train 생략) 기존 모델 Weight 기반으로 Test만 수행 |
| **11~12** | CASE 모델 Inference (Pretrained weights 사용) |
| **13** | outputs/ 결과 중 CSV만 복사 (`outputs_csv/`) |
| **14** | Fold 기준으로 결과 요약 (`outputs_csv_summaries/`) |
| **15** | 정상군 리스트 기반 group 컬럼(0=Normal,1=Others) 추가 |
| **16** | Demography 설문 전처리 → `Demo_processed.csv` 생성 |
| **17** | 설문 + 모델 요약 결과 병합 (`data/final_data/`) |
| **18** | 최종 ML 모델 추론 (`model/18_inference.py`) |

---

## Key Components

- **MMACTION2 (SlowFast)**  
  비디오 행동 피처 추출 backbone.  
  mmcv + torch 1.12.1 기반, GPU CUDA 11.3 환경에서 구동.

- **CASE Model**  
  Self-supervised + Contrastive 기반 비디오 representation 모델.  
  이미 학습된 weight를 기반으로 Inference만 수행.

- **Final ML Model**  
  GradientBoosting(학습된 weight) / XGBoost / LogisticRegression 등  
  Tabular + Video feature 기반 ASD 예측 수행.

---

## Run Inference Only (Pretrained Models)

실행:
```bash
bash run_pipeline.sh
```

출력 결과:
```
./Prediction_Results/inference_results_all_folds.csv
```

---

## Output Example

```| fold | patient_id | pred_numeric | pred_label | prob_others | prob_normal | ground_truth |
|------|-------------|-------------|--------------|--------------|----------------|
| 0 | patient_1 | 0 | Normal | 0.18 | 0.82 | Normal |
| 0 | patient_2 | 1 | Others | 0.75 | 0.25 | Others |
| 1 | patient_3 | 0 | Normal | 0.32 | 0.68 | Normal |
```

---

## Environment Summary

| Component | Version |
|------------|----------|
| Python | 3.8.20 |
| CUDA | 11.3 |
| PyTorch | 1.12.1+cu113 |
| TorchVision | 0.13.1+cu113 |
| MMCV | 2.0.0 |
| MMACTION2 | Local Editable |
| scikit-learn | 1.3.2 |
| pandas | ≥1.3 |
| numpy | ≥1.21 |
| seaborn / matplotlib / shap | Latest |
