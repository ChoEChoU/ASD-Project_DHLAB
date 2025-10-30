# model/main-matching_Grid.py
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import xgboost as xgb
import os
import warnings
import argparse

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--rfe', type=int, default=0, help='RFE 선택 수 (0이면 사용 안 함)')
parser.add_argument('--grid', action='store_true', help='GridSearch 사용 여부')
parser.add_argument('--metric', type=str, default='f1', choices=['f1', 'auc'], help='성능 기준(f1/auc)')
parser.add_argument('--impute', type=str, default='zero', choices=['zero', 'growing'], help='결측치 처리 방식')
# 아래 3개는 원하면 바꿔 쓸 수 있게 인자로도 노출
parser.add_argument('--data_dir', type=str, default='./data/final_data', help='fold 병합 최종 데이터 경로(17단계 출력)')
parser.add_argument('--save_root', type=str, default='./model_results_matching', help='실험 결과 저장 루트')
parser.add_argument('--group_name', type=str, default='Matched_E_Cell_1017', help='결과 폴더 표시용 그룹명')
parser.add_argument('--modality', type=str, default='multimodal/video_demo', help='결과 폴더 표시용 모달리티명')
parser.add_argument('--n_splits', type=int, default=5)
args = parser.parse_args()

rfe_n_features   = args.rfe if args.rfe > 0 else None
use_grid         = args.grid
selection_metric = args.metric            # 'f1' or 'auc'
impute_mode      = args.impute            # 'zero' or 'growing'
data_dir         = args.data_dir
save_root        = args.save_root
group_name       = args.group_name
modality_name    = args.modality
n_splits         = args.n_splits

os.environ["OMP_NUM_THREADS"]       = "4"
os.environ["OPENBLAS_NUM_THREADS"]  = "4"
os.environ["MKL_NUM_THREADS"]       = "4"
os.environ["VECLIB_MAXIMUM_THREADS"]= "4"
os.environ["NUMEXPR_NUM_THREADS"]   = "4"

# ------------------------------
# 모델/그리드 정의
# ------------------------------
param_grid_dict = {
    'RandomForest': {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'max_features': ['sqrt', 'log2', 0.5],
    },
    'LogisticRegression': {
        'penalty': ['l2'],
        'C': [0.01, 0.1, 1.0],
    },
    'SVC': {
        'kernel': ['linear', 'rbf'],
        'C': [0.01, 0.1, 1.0],
        'gamma': ['scale', 0.1, 0.01],
    },
    'GradientBoosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.03, 0.05, 0.1],
        'max_depth': [2, 3],
        'min_samples_leaf': [1, 5, 10],
        'subsample': [0.6, 0.8, 1.0],
    },
    'KNN': {
        'n_neighbors': [5, 7, 11, 15],
        'weights': ['uniform', 'distance'],
    },
    'XGBoost': {
        'n_estimators': [200, 400],
        'learning_rate': [0.03, 0.05, 0.1],
        'max_depth': [3, 5],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_lambda': [0, 1, 5],
        'reg_alpha': [0, 0.1],
    }
}

model_base = {
    'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'SVC': SVC(probability=True, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
}

# ------------------------------
# 보간 함수 (row-wise growing average)
# ------------------------------
def growing_average_impute_rowwise(df, target_cols, default_fill_zero=True, prefill_value=None):
    df = df.copy()
    result_df = pd.DataFrame(index=df.index, columns=target_cols)

    for idx, row in df[target_cols].iterrows():
        values = row.values.astype(float)
        mask = ~np.isnan(values)

        running_sum = 0.0
        running_count = 0
        imputed = []

        # prefill
        if default_fill_zero:
            fill_val = 0.0
        elif prefill_value is not None:
            fill_val = prefill_value
        else:
            valid_vals = values[mask]
            fill_val = valid_vals.mean() if len(valid_vals) > 0 else 0.0

        for v in values:
            if not np.isnan(v):
                running_sum += v
                running_count += 1
                imputed.append(v)
            else:
                if running_count == 0:
                    imputed.append(fill_val)
                else:
                    imputed.append(running_sum / running_count)

        result_df.loc[idx] = imputed

    return result_df

# 월 순서 정렬용 도우미: 접두부가 "02_", "04_"처럼 되어있으면 그 숫자로 정렬
def _extract_month_leading(col_name: str) -> int:
    head = col_name.split('_')[0]
    return int(head) if head.isdigit() else 999

# ------------------------------
# 저장 기본 경로
# ------------------------------
condition_str = f"{selection_metric}"
condition_str += f"__RFE{rfe_n_features}" if rfe_n_features else "__noRFE"
condition_str += f"__grid" if use_grid else "__noGrid"
condition_str += f"__{impute_mode}"

save_dir = os.path.join(
    save_root,
    modality_name.replace('/', '_'),
    group_name,
    condition_str
)
os.makedirs(save_dir, exist_ok=True)

all_results = []

print(f"▶ DATA DIR   : {data_dir}")
print(f"▶ SAVE DIR   : {save_dir}")
print(f"▶ CONDITION  : {condition_str}")

for i in range(n_splits):
    print(f"\n>>> Processing fold = {i}")

    train_path = os.path.join(data_dir, f"train_fold{i}.csv")
    valid_path = os.path.join(data_dir, f"valid_fold{i}.csv")
    test_path  = os.path.join(data_dir, f"test_fold{i}.csv")

    # --------------------------
    # 데이터 로드
    # --------------------------
    train_data = pd.read_csv(train_path)
    valid_data = pd.read_csv(valid_path)
    test_data  = pd.read_csv(test_path)

    # 안전 장치: 문자열로 통일
    for df in (train_data, valid_data, test_data):
        if 'patient_id' in df.columns:
            df['patient_id'] = df['patient_id'].astype(str)

    # CASE 파이프라인 상 group은 0/1 (0=Normal,1=Others)로 이미 존재
    # 혹시 문자열 라벨이 들어오면 매핑
    def _to_int_label(s):
        if s.dtype == object:
            if set(s.unique()) <= {'normal', 'other_e', 'Normal', 'Others'}:
                m = {'normal':0, 'Normal':0, 'other_e':1, 'Others':1}
                return s.map(m).astype(int)
        return s.astype(int)

    # --------------------------
    # 학습용 테이블 구성
    # - patient_id / group 제외 전체를 feature로 사용
    # - growing 모드면 *_prob_class_0 만 보간 적용
    # --------------------------
    for df in (train_data, valid_data, test_data):
        if 'subjectId' in df.columns:  # 혹시 남아있다면 patient_id로 통일
            df.rename(columns={'subjectId':'patient_id'}, inplace=True)

    y_train = _to_int_label(train_data['group'])
    y_valid = _to_int_label(valid_data['group'])
    y_test  = _to_int_label(test_data['group'])

    # 특징 컬럼
    drop_cols = [c for c in ['patient_id', 'group'] if c in train_data.columns]
    feature_cols_all = [c for c in train_data.columns if c not in drop_cols]

    X_train = train_data[feature_cols_all].copy()
    X_valid = valid_data[feature_cols_all].copy()
    X_test  = test_data[feature_cols_all].copy()

    # growing average는 비디오 점수열만 대상으로 수행
    if impute_mode == 'growing':
        class0_cols = [c for c in feature_cols_all if c.endswith('prob_class_0')]
        class0_cols_sorted = sorted(class0_cols, key=_extract_month_leading)
        if class0_cols_sorted:
            X_train[class0_cols_sorted] = growing_average_impute_rowwise(X_train, class0_cols_sorted)
            X_valid[class0_cols_sorted] = growing_average_impute_rowwise(X_valid, class0_cols_sorted)
            X_test[class0_cols_sorted]  = growing_average_impute_rowwise(X_test,  class0_cols_sorted)

    # 남은 NaN은 0으로
    X_train = X_train.fillna(0)
    X_valid = X_valid.fillna(0)
    X_test  = X_test.fillna(0)

    # 열 정합 강제
    feature_cols = X_train.columns
    X_valid = X_valid.reindex(columns=feature_cols, fill_value=0)
    X_test  = X_test.reindex(columns=feature_cols,  fill_value=0)

    # RFE(옵션)
    if rfe_n_features:
        print(f"  ▶ RFE selecting top {rfe_n_features} features using LogisticRegression")
        rfe_selector_model = LogisticRegression(max_iter=1000, random_state=42)
        rfe = RFE(estimator=rfe_selector_model, n_features_to_select=rfe_n_features)
        rfe.fit(X_train, y_train)
        selected_features = list(X_train.columns[rfe.support_])
        X_train = X_train[selected_features]
        X_valid = X_valid[selected_features]
        X_test  = X_test[selected_features]
        feature_cols = selected_features
        print(f"    - Feature count after RFE: {len(feature_cols)}")

    # 스케일링
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_valid = pd.DataFrame(scaler.transform(X_valid), columns=feature_cols, index=X_valid.index)
    X_test  = pd.DataFrame(scaler.transform(X_test),  columns=feature_cols, index=X_test.index)

    # train+valid를 최종학습에 사용
    X_full = pd.concat([X_train, X_valid], axis=0)
    y_full = pd.concat([y_train, y_valid], axis=0)

    best_score   = -1
    best_model   = None
    best_name    = ""
    best_thresh  = 0.5
    best_val_auc = 0.0

    # --------------------------
    # 모델 루프
    # --------------------------
    for m_name, base in model_base.items():
        print(f"  ▶ Training/Evaluating: {m_name}")

        scoring_metric = 'roc_auc' if selection_metric == 'auc' else 'f1'
        model = clone(base)
        grid_params = param_grid_dict.get(m_name, {})

        if use_grid and grid_params:
            print(f"    - GridSearchCV(scoring={scoring_metric})")
            grid = GridSearchCV(model, grid_params, scoring=scoring_metric, cv=3, n_jobs=-1)
            grid.fit(X_train, y_train)
            validation_model = grid.best_estimator_
        else:
            validation_model = model
            validation_model.fit(X_train, y_train)

        # proba
        if hasattr(validation_model, "predict_proba"):
            val_proba = validation_model.predict_proba(X_valid)[:, 1]
        else:
            # 확률 못 내는 모델은 결정함수로 대체(선형 SVC 등). 없으면 0.
            if hasattr(validation_model, "decision_function"):
                z = validation_model.decision_function(X_valid)
                # min-max로 [0,1] 정규화
                val_proba = (z - z.min()) / (z.max() - z.min() + 1e-9)
            else:
                val_proba = np.zeros_like(y_valid, dtype=float)

        # F1 튜닝 (threshold sweep)
        best_f1_tmp, best_t = 0.0, 0.5
        for t in np.linspace(0.1, 0.9, 17):
            val_pred = (val_proba >= t).astype(int)
            f1t = f1_score(y_valid, val_pred, zero_division=0)
            if f1t > best_f1_tmp:
                best_f1_tmp, best_t = f1t, t

        # 기본 임계값 예측도 같이 계산
        if hasattr(validation_model, "predict"):
            val_pred_default = validation_model.predict(X_valid)
        else:
            val_pred_default = (val_proba >= 0.5).astype(int)

        val_f1  = f1_score(y_valid, val_pred_default, zero_division=0)
        try:
            val_auc = roc_auc_score(y_valid, val_proba)
        except:
            val_auc = np.nan

        score_for_select = best_f1_tmp if selection_metric == 'f1' else val_auc
        print(f"    - Valid AUC: {val_auc:.4f}, Valid F1(default): {val_f1:.4f}, F1(tuned): {best_f1_tmp:.4f}")

        if score_for_select > best_score:
            best_score   = score_for_select
            best_model   = clone(validation_model)
            best_name    = m_name
            best_thresh  = best_t
            best_val_auc = val_auc

    # --------------------------
    # Final train & Test with best model
    # --------------------------
    print(f"  ▶ Best model: {best_name} (metric={selection_metric})")
    best_model.fit(X_full, y_full)

    if hasattr(best_model, "predict_proba"):
        test_proba = best_model.predict_proba(X_test)[:, 1]
    else:
        if hasattr(best_model, "decision_function"):
            zt = best_model.decision_function(X_test)
            test_proba = (zt - zt.min()) / (zt.max() - zt.min() + 1e-9)
        else:
            test_proba = np.zeros_like(y_test, dtype=float)

    test_pred = (test_proba >= best_thresh).astype(int)

    acc  = accuracy_score(y_test, test_pred)
    try:
        auroc = roc_auc_score(y_test, test_proba)
    except:
        auroc = np.nan
    prec = precision_score(y_test, test_pred, zero_division=0)
    rec  = recall_score(y_test, test_pred, zero_division=0)
    f1v  = f1_score(y_test, test_pred, zero_division=0)

    # 혼동행렬 저장
    cm = confusion_matrix(y_test, test_pred, labels=[0,1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{group_name} -- fold{i} | {best_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    cm_path = os.path.join(save_dir, f'{group_name}__fold{i}_{best_name}_cm.png')
    plt.savefig(cm_path, bbox_inches='tight', dpi=200)
    plt.close()

    # SHAP 저장 (가능하면)
    print("  ▶ Calculating SHAP...")
    try:
        if hasattr(best_model, "feature_importances_"):
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer(X_test)
        else:
            explainer = shap.Explainer(lambda x: best_model.predict_proba(x), X_full)
            shap_values = explainer(X_test)

        if isinstance(shap_values, shap.Explanation):
            shap_to_plot = shap_values
            shap_arr = shap_values.values
            X_plot = getattr(shap_values, "data", X_test)
            feat_names = getattr(shap_values, "feature_names", list(X_test.columns))
        else:
            shap_arr = shap_values
            X_plot = X_test
            feat_names = list(X_test.columns)
            shap_to_plot = shap.Explanation(values=shap_arr, data=X_plot, feature_names=feat_names)

        # 클래스 축이 마지막에 있는 경우(class=2) → class1만 사용
        if isinstance(shap_to_plot.values, np.ndarray) and shap_to_plot.values.ndim == 3 and shap_to_plot.values.shape[2] == 2:
            shap_to_plot = shap.Explanation(
                values=shap_to_plot.values[:, :, 1],
                base_values=np.array(shap_to_plot.base_values)[:, 1] if np.array(shap_to_plot.base_values).ndim==2 else shap_to_plot.base_values,
                data=shap_to_plot.data,
                feature_names=shap_to_plot.feature_names
            )

        # beeswarm
        plt.figure()
        shap.plots.beeswarm(shap_to_plot, max_display=20, show=False)
        shap_path = os.path.join(save_dir, f'{group_name}__fold{i}_{best_name}_shap_beeswarm.png')
        plt.savefig(shap_path, bbox_inches='tight', dpi=300)
        plt.close()

        # 평균 절대 SHAP
        shap_mean = np.mean(np.abs(np.array(shap_to_plot.values)), axis=0)
        shap_df = pd.DataFrame({'feature': feat_names, 'mean_abs_shap': shap_mean})
        shap_df.sort_values('mean_abs_shap', ascending=False, inplace=True)
        shap_df.to_csv(os.path.join(save_dir, f'{group_name}__fold{i}_{best_name}_shap_values.csv'), index=False)

    except Exception as e:
        print(f"    - SHAP failed: {e}")

    cm_flat = cm.flatten()
    cm_dict = {f'cm_{k}': v for k, v in enumerate(cm_flat)} if len(cm_flat)==4 else {f'cm_{k}':0 for k in range(4)}

    all_results.append({
        'group': group_name, 'modality': modality_name, 'fold': i, 'model': best_name,
        'val_auc': best_val_auc, 'test_acc': acc, 'test_auroc': auroc,
        'test_precision': prec, 'test_recall': rec, 'test_f1': f1v, **cm_dict
    })

    # 예측치 저장
    pred_df = pd.DataFrame({'true_label': y_test, 'pred_label': test_pred, 'pred_proba': test_proba})
    pred_df.to_csv(os.path.join(save_dir, f'{group_name}__fold{i}_{best_name}_predictions.csv'), index=False, encoding='utf-8-sig')

    # 사용한 특성 리스트 저장
    pd.Series(feature_cols).to_csv(os.path.join(save_dir, f'{group_name}__fold{i}_{best_name}_features_used.csv'), index=False)

# --------------------------
# Fold 결과 + 평균 결과 저장
# --------------------------
result_df = pd.DataFrame(all_results)

mean_results = (
    result_df
    .groupby(['group', 'modality', 'model'], as_index=False)
    .agg({
        'val_auc': 'mean',
        'test_acc': 'mean', 'test_auroc': 'mean',
        'test_precision': 'mean', 'test_recall': 'mean', 'test_f1': 'mean',
        'cm_0':'mean', 'cm_1':'mean', 'cm_2':'mean', 'cm_3':'mean'
    })
)
mean_results['fold'] = -1

final_df = pd.concat([result_df, mean_results[['group','modality','model','val_auc','test_acc','test_auroc','test_precision','test_recall','test_f1','cm_0','cm_1','cm_2','cm_3','fold']]], ignore_index=True)
final_path = os.path.join(save_dir, 'final_results.csv')
final_df.to_csv(final_path, index=False, encoding='utf-8-sig')

print(f"\n✅ 저장 완료: {final_path}")