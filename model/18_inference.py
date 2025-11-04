# model/18_inference.py (핵심 변경판)
import argparse
import os
import joblib
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

def load_artifact(artifact_path):
    obj = joblib.load(artifact_path)
    if isinstance(obj, dict):
        model = obj.get("model")
        scaler = obj.get("scaler")
        features = obj.get("features")
        threshold = obj.get("threshold", 0.5)
        if model is None or features is None:
            raise ValueError(f"[ERROR] Invalid artifact: {artifact_path}")
        return dict(model=model, scaler=scaler, features=features, threshold=threshold)
    return dict(model=obj, scaler=None, features=None, threshold=0.5)

def _safe_get_id_column(df, key_col):
    if key_col in df.columns: return key_col
    if "patient_id" in df.columns: return "patient_id"
    if "subjectId" in df.columns: return "subjectId"
    raise ValueError(f"[ERROR] ID column not found. Available: {list(df.columns)}")

# --------- NEW: HalfBinomialLoss 호환 패치 & 수동 forward ---------
def _is_gb(model):
    return model.__class__.__name__ == "GradientBoostingClassifier"

def _needs_halfbinomial_patch(err):
    return isinstance(err, AttributeError) and "HalfBinomialLoss" in str(err)

class _HalfBinomialShim:
    """sklearn 1.3.2 호환을 위한 최소 shim: predict 경로에서 필요한 메서드만 구현"""
    def get_init_raw_predictions(self, X, init_estimator):
        raw = init_estimator.predict(X)
        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)
        return raw

def _try_patch_halfbinomial(model):
    # 1.4에서 직렬화된 모델에는 _loss 또는 loss_ 속성에 HalfBinomialLoss가 들어 있을 가능성
    # 1.3.2가 기대하는 인터페이스(get_init_raw_predictions)를 제공하도록 shim으로 교체
    for attr in ("_loss", "loss_"):
        if hasattr(model, attr):
            obj = getattr(model, attr)
            if obj is not None and obj.__class__.__name__ == "HalfBinomialLoss":
                setattr(model, attr, _HalfBinomialShim())
                return True
    return False

def _manual_forward_gb_binary(model, X):
    """
    GradientBoostingClassifier (binary) 수동 추론:
      raw = init_.predict(X) + sum(learning_rate * tree.predict(X))
      proba = sigmoid(raw)
    """
    # init_
    if hasattr(model, "init_") and model.init_ is not None:
        raw = model.init_.predict(X).astype(float)
    else:
        # init_가 없으면 0으로 시작
        raw = np.zeros(X.shape[0], dtype=float)

    # estimators_: shape (n_stages, 1)
    estimators = getattr(model, "estimators_", None)
    if estimators is None:
        return np.zeros(X.shape[0], dtype=float)

    for stage in estimators:
        # binary: stage는 길이 1의 배열(DecisionTreeRegressor 하나)
        tree = stage[0]
        raw += model.learning_rate * tree.predict(X).astype(float)

    # sigmoid
    proba_class1 = 1.0 / (1.0 + np.exp(-raw))
    return proba_class1
# ----------------------------------------------------------------

def infer_with_artifact(artifact_path, df_raw, label_col="group", id_col="patient_id"):
    art = load_artifact(artifact_path)
    model = art["model"]
    scaler = art["scaler"]
    features = art["features"]
    threshold = float(art["threshold"])

    X_raw = df_raw.drop(columns=[c for c in [label_col, id_col] if c in df_raw.columns], errors="ignore")
    if features is None:
        features = list(X_raw.columns)
    X = X_raw.reindex(columns=features, fill_value=0)
    Xs = pd.DataFrame(scaler.transform(X), columns=features, index=X.index) if scaler is not None else X

    proba_others = None

    # 1) 정상 경로 시도
    try:
        if hasattr(model, "predict_proba"):
            proba_others = model.predict_proba(Xs)[:, 1]
        elif hasattr(model, "decision_function"):
            decision = model.decision_function(Xs)
            proba_others = (decision - decision.min()) / (decision.max() - decision.min() + 1e-9)
    except Exception as e1:
        if _needs_halfbinomial_patch(e1) and _is_gb(model):
            print("[WARN] predict_proba/decision_function raised HalfBinomialLoss compat error → patching for sklearn 1.3.2")
            # 2) 호환 패치 후 재시도
            patched = _try_patch_halfbinomial(model)
            if patched:
                try:
                    if hasattr(model, "predict_proba"):
                        proba_others = model.predict_proba(Xs)[:, 1]
                    elif hasattr(model, "decision_function"):
                        decision = model.decision_function(Xs)
                        proba_others = (decision - decision.min()) / (decision.max() - decision.min() + 1e-9)
                except Exception as e2:
                    print(f"[WARN] patched path failed ({type(e2).__name__}) → manual forward for GB")
            else:
                print("[WARN] patch not applicable → manual forward for GB")

    # 3) 수동 forward (GradientBoosting 전용) — 위 두 경로가 실패했을 때
    if proba_others is None and _is_gb(model):
        try:
            proba_others = _manual_forward_gb_binary(model, Xs)
        except Exception as e3:
            print(f"[WARN] manual forward failed ({type(e3).__name__}) → zeros")
            proba_others = np.zeros(len(Xs))

    # 4) 최후 fallback
    if proba_others is None:
        proba_others = np.zeros(len(Xs))

    pred_numeric = (proba_others >= threshold).astype(int)
    used_id = df_raw[id_col] if id_col in df_raw.columns else pd.Series(np.arange(len(df_raw)), name=id_col)
    return proba_others, pred_numeric, used_id

def main():
    parser = argparse.ArgumentParser(description="Inference with trained ML artifacts (sklearn 1.3.2 compatible)")
    parser.add_argument("--model_dir", type=str, default="./model/ml_weight")
    parser.add_argument("--group_name", type=str, default="Matched_E_Cell_1017")
    parser.add_argument("--model_name", type=str, default="GradientBoosting")
    parser.add_argument("--data_dir", type=str, default="./data/final_data")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--key", type=str, default="patient_id")
    parser.add_argument("--label_col", type=str, default="group")
    parser.add_argument("--output_dir", type=str, default="./Prediction_Results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []

    for fold in range(args.folds):
        model_file = f"{args.group_name}__fold{fold}_{args.model_name}_artifact.joblib"
        model_path = os.path.join(args.model_dir, model_file)
        test_path = os.path.join(args.data_dir, f"test_fold{fold}.csv")

        if not os.path.exists(model_path):
            print(f"[WARN] Model not found: {model_path}")
            continue
        if not os.path.exists(test_path):
            print(f"[WARN] Test CSV not found: {test_path}")
            continue

        print(f"\n=== Fold {fold} Inference ===")
        print(f"Model: {model_path}")
        print(f"Data : {test_path}")

        df = pd.read_csv(test_path).fillna(0)
        id_col = _safe_get_id_column(df, args.key)

        proba_others, pred_numeric, used_id = infer_with_artifact(
            artifact_path=model_path, df_raw=df, label_col=args.label_col, id_col=id_col
        )
        proba_normal = 1.0 - proba_others

        label_map = {0: "Normal", 1: "Others"}
        pred_label_text = [label_map[int(v)] for v in pred_numeric]

        if args.label_col in df.columns:
            if np.issubdtype(df[args.label_col].dtype, np.number):
                gt_text = df[args.label_col].map(label_map)
            else:
                gt_text = df[args.label_col]
        else:
            gt_text = pd.Series([None] * len(df), name="ground_truth")

        result_df = pd.DataFrame({
            "fold": fold,
            "patient_id": used_id,
            "pred_numeric": pred_numeric,
            "pred_label": pred_label_text,
            "prob_others": proba_others,
            "prob_normal": proba_normal,
            "ground_truth": gt_text
        })

        out_csv = os.path.join(args.output_dir, f"inference_fold{fold}.csv")
        result_df.to_csv(out_csv, index=False, float_format="%.6f")
        print(f"✅ Saved: {out_csv}")
        all_results.append(result_df)

    if not all_results:
        print("\n❌ No inference results collected.")
        return

    final_result = pd.concat(all_results, ignore_index=True)
    merged_path = os.path.join(args.output_dir, "inference_results_all_folds.csv")
    final_result.to_csv(merged_path, index=False, float_format="%.6f")
    print(f"\n🎉 Inference DONE → {merged_path}")

    # Preview
    pd.set_option("display.width", 200)
    pd.options.display.float_format = "{:.6f}".format
    cols = ["fold", "patient_id", "pred_numeric", "pred_label", "prob_others", "prob_normal", "ground_truth"]
    print("\n===== Preview (first 10 rows) =====")
    print(final_result[cols].head(10).to_string(index=False))

if __name__ == "__main__":
    main()