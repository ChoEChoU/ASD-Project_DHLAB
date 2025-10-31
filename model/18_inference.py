# model/18_inference.py
import argparse
import os
import joblib
import pandas as pd
import numpy as np


def load_artifact(artifact_path):
    """joblib으로 저장된 모델 아티팩트 로드"""
    obj = joblib.load(artifact_path)

    # Dict 형태(권장)
    if isinstance(obj, dict):
        model = obj.get("model")
        scaler = obj.get("scaler")
        features = obj.get("features")
        threshold = obj.get("threshold", 0.5)

        if model is None or features is None:
            raise ValueError(f"[ERROR] Invalid artifact: {artifact_path}")
        return dict(model=model, scaler=scaler, features=features, threshold=threshold)

    # 모델 단독 저장 형태
    return dict(model=obj, scaler=None, features=None, threshold=0.5)


def _safe_get_id_column(df, key_col):
    """patient_id 또는 subjectId 자동 인식"""
    if key_col in df.columns:
        return key_col
    if "patient_id" in df.columns:
        return "patient_id"
    if "subjectId" in df.columns:
        return "subjectId"
    raise ValueError(f"[ERROR] ID column not found. Available: {list(df.columns)}")


def infer_with_artifact(artifact_path, df_raw, label_col="group", id_col="patient_id"):
    art = load_artifact(artifact_path)
    model = art["model"]
    scaler = art["scaler"]
    features = art["features"]
    threshold = float(art["threshold"])

    # 불필요한 열 제거
    X_raw = df_raw.drop(columns=[c for c in [label_col, id_col] if c in df_raw.columns], errors="ignore")

    # feature 순서 보정
    if features is None:
        features = list(X_raw.columns)
    X = X_raw.reindex(columns=features, fill_value=0)

    # 스케일러 적용
    if scaler is not None:
        Xs = pd.DataFrame(scaler.transform(X), columns=features, index=X.index)
    else:
        Xs = X

    # 예측 확률 계산
    try:
        if hasattr(model, "predict_proba"):
            proba_others = model.predict_proba(Xs)[:, 1]
        elif hasattr(model, "decision_function"):
            decision = model.decision_function(Xs)
            proba_others = (decision - decision.min()) / (decision.max() - decision.min() + 1e-9)
        else:
            proba_others = np.zeros(len(Xs))
    except Exception as e:
        print(f"[WARN] predict_proba failed ({type(e).__name__}) → fallback to zeros")
        proba_others = np.zeros(len(Xs))

    pred_numeric = (proba_others >= threshold).astype(int)
    used_id = df_raw[id_col] if id_col in df_raw.columns else pd.Series(np.arange(len(df_raw)), name=id_col)

    return proba_others, pred_numeric, used_id


def main():
    parser = argparse.ArgumentParser(description="Inference with trained ML artifacts (sklearn 1.3.2 compatible)")
    parser.add_argument("--model_dir", type=str, default="./model/ml_weight", help="학습된 .joblib 아티팩트 폴더")
    parser.add_argument("--group_name", type=str, default="Matched_E_Cell_1017", help="모델 파일 prefix (Group명)")
    parser.add_argument("--model_name", type=str, default="GradientBoosting", help="모델 이름 (예: GradientBoosting)")
    parser.add_argument("--data_dir", type=str, default="./data/final_data", help="fold별 test CSV 경로")
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
            artifact_path=model_path,
            df_raw=df,
            label_col=args.label_col,
            id_col=id_col
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
        result_df.to_csv(out_csv, index=False)
        print(f"✅ Saved: {out_csv}")
        all_results.append(result_df)

    if not all_results:
        print("\n❌ No inference results collected.")
        return

    final_result = pd.concat(all_results, ignore_index=True)
    merged_path = os.path.join(args.output_dir, "inference_results_all_folds.csv")
    final_result.to_csv(merged_path, index=False)
    print(f"\n🎉 Inference DONE → {merged_path}")
    print(final_result.head())


if __name__ == "__main__":
    main()