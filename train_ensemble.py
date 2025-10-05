# train_ensemble.py
import json, joblib, numpy as np, pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# ---- project imports
from schema_map import to_canonical, CANON_KEYS
from data_prep import engineer_features

BASE = Path(__file__).resolve().parent
OUT  = BASE / "models" / "ensemble"
OUT.mkdir(parents=True, exist_ok=True)

# 1) Load Kepler + (optional) TESS TOI
kpath = BASE / "data" / "kepler_cumulative.csv"
tpath = BASE / "data" / "TOI_2025.10.04_21.07.34.csv"  # replace with your TESS CSV

frames = []
if kpath.exists():
    frames.append(pd.read_csv(kpath, low_memory=False).assign(__src="kepler"))
if tpath.exists():
    frames.append(pd.read_csv(tpath, low_memory=False).assign(__src="tess"))
df_raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
assert not df_raw.empty, "No training data found."

# 2) Canonicalize + engineer features
canon = to_canonical(df_raw, source=None)
canon = engineer_features(canon)

# 3) Pick features/labels
feat_cols = [c for c in canon.columns if c in CANON_KEYS and c != "label"]
# include engineered ones if present in your current model
for extra in ["depth_vs_ror2_abs","duration_over_period","duration_over_aRs"]:
    if extra in canon.columns:
        feat_cols.append(extra)

X = canon[feat_cols].copy()
y = canon["label"].astype(int)

# 4) Preprocess (impute + scale for linear models); tree models ignore scaling
pre = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc", StandardScaler(with_mean=True, with_std=True))
])

# 5) Base learners
learners = []
if HAS_XGB:
    learners.append(("xgb", XGBClassifier(
        n_estimators=400, max_depth=6, subsample=0.7, colsample_bytree=0.8,
        learning_rate=0.05, reg_lambda=1.0, eval_metric="mlogloss",
        n_jobs=-1, tree_method="hist"
    )))
learners.append(("rf", RandomForestClassifier(
    n_estimators=600, max_depth=None, min_samples_leaf=2, n_jobs=-1, class_weight="balanced_subsample"
)))
learners.append(("lr", LogisticRegression(
    max_iter=2000, multi_class="multinomial", C=1.0
)))

# 6) Stacking (meta-learner = calibrated logistic regression)
stack = StackingClassifier(
    estimators=learners,
    final_estimator=LogisticRegression(max_iter=2000, multi_class="multinomial"),
    passthrough=True
)

# 7) Full pipeline: preprocess -> stack -> calibrate (isotonic)
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
pre.fit(Xtr, ytr)
Xtr_p = pre.transform(Xtr)
Xva_p = pre.transform(Xva)

stack.fit(Xtr_p, ytr)
cal = CalibratedClassifierCV(stack, method="isotonic", cv=5)
cal.fit(Xva_p, yva)

# 8) Save artifacts
joblib.dump(pre, OUT / "preproc.pkl")
joblib.dump(cal, OUT / "model.pkl")
meta = {
    "features": feat_cols,
    "labels": {0: "FALSE POSITIVE", 1: "CANDIDATE", 2: "CONFIRMED"},
    "source_mix": df_raw["__src"].value_counts().to_dict(),
    "note": "Stacked ensemble (XGB+RF+LR) with isotonic calibration."
}
(OUT / "metadata.json").write_text(json.dumps(meta, indent=2))
print("Saved to:", OUT)
