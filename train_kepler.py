# train_kepler.py
import argparse
import os
import sys
import csv
import json
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from data_prep import prepare_training_matrices
from utils import save_model


def read_kepler_csv(path: str) -> pd.DataFrame:
    """Try a strict fast read first; fall back to tolerant parser if needed."""
    try:
        return pd.read_csv(
            path,
            sep=",",
            engine="c",
            low_memory=False,
            encoding="utf-8",
            quoting=csv.QUOTE_MINIMAL,
        )
    except Exception as e:
        print(f"[warn] Strict CSV parse failed: {e}", file=sys.stderr)
        print("[info] Falling back to engine='python' with delimiter sniffing and on_bad_lines='warn'...", file=sys.stderr)
        return pd.read_csv(
            path,
            sep=None,              # auto-detect delimiter
            engine="python",       # required for sep=None
            low_memory=False,
            encoding="utf-8",
            on_bad_lines="warn",   # change to "skip" to silence warnings
            quoting=csv.QUOTE_MINIMAL,
        )


def main(args):
    # --- Load data robustly ---
    df = read_kepler_csv(args.csv)
    print(f"[info] Loaded data: shape={df.shape}")

    # --- Prep data (your function) ---
    X_train, X_test, y_train, y_test, feat_cols, class_weight_dict = prepare_training_matrices(df)

    # --- Encode string labels -> integers for XGBoost ---
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    print(f"[info] Classes: {list(le.classes_)}")

    # --- Model ---
    model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        max_depth=7,
        learning_rate=0.07,
        subsample=0.9,
        colsample_bytree=0.9,
        n_estimators=800,
        reg_lambda=1.5,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        num_class=len(le.classes_)  # defensive; XGB will infer but this is explicit
    )

    # Optional: apply class weights as sample weights if provided
    sample_weight = None
    if class_weight_dict:
        # class_weight_dict is presumed to be mapping from ORIGINAL labels (strings) to weights
        try:
            sample_weight = y_train.map(class_weight_dict).to_numpy()
            print("[info] Using sample weights from class_weight_dict.")
        except Exception as e:
            print(f"[warn] Could not build sample weights from class_weight_dict: {e}", file=sys.stderr)

    # --- Fit ---
    model.fit(X_train, y_train_enc, sample_weight=sample_weight)

    # --- Predict (decode back to original string labels for reports) ---
    y_pred_enc = model.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)

    # --- Reports ---
    print("=== Classification report (test) ===")
    report = classification_report(y_test, y_pred, digits=3, labels=le.classes_)
    print(report)

    print("=== Confusion matrix (normalized by true) ===")
    cm = confusion_matrix(y_test, y_pred, labels=le.classes_, normalize="true")
    print(cm)

    # --- Save ---
    os.makedirs(args.out_dir, exist_ok=True)

    # Save model + metadata; persist label mapping for future inference
    meta = {
        "features": feat_cols,
        "class_weight": class_weight_dict,
        "model_type": "XGBClassifier",
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "label_classes": list(le.classes_),                 # index -> class label order
        "label_to_index": {c: int(i) for i, c in enumerate(le.classes_)},
    }
    save_model(model, args.out_dir, feat_cols, metadata=meta)

    # Also drop a text report
    with open(os.path.join(args.out_dir, "last_report.txt"), "w") as f:
        f.write(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to Kepler cumulative CSV")
    parser.add_argument("--out_dir", default="models", help="Output directory for model artifacts")
    args = parser.parse_args()
    main(args)
