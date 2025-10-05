# train_ensemble.py
import json, joblib, numpy as np, pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

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

# -------------------------------
# 1) Load Kepler + (optional) TESS
# -------------------------------
kpath = BASE / "data" / "kepler_cumulative.csv"

# Try the exact TESS file; if not present, pick the newest TOI*.csv in /data
tpath = BASE / "data" / "TOI_2025.10.04_21.07.34.csv"
if not tpath.exists():
    cand = sorted((BASE / "data").glob("TOI*.csv"))
    tpath = cand[-1] if cand else None

frames = []
if kpath.exists():
    print(f"[load] Kepler CSV: {kpath}")
    frames.append(pd.read_csv(kpath, low_memory=False).assign(__src="kepler"))
else:
    print(f"[load] Kepler CSV missing at: {kpath}")

if tpath and Path(tpath).exists():
    print(f"[load] TESS CSV:   {tpath}")
    frames.append(pd.read_csv(tpath, low_memory=False).assign(__src="tess"))
else:
    print(f"[load] TESS CSV not found. Skipping.")

df_raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
if df_raw.empty:
    raise FileNotFoundError(
        "No training data found. Looked for:\n"
        f" - {kpath}\n"
        f" - {tpath if tpath else '(no TOI*.csv found)'}\n"
        "Put your files under <repo>/data/ and re-run."
    )
print(f"[load] Combined rows: {len(df_raw):,}")

# ----------------------------------------
# 2) Canonicalize + engineer + pick columns
# ----------------------------------------
canon = to_canonical(df_raw, source=None)
canon = engineer_features(canon)

# Feature list: canonical + engineered (if present)
feat_cols = [c for c in canon.columns if c in CANON_KEYS and c != "label"]
for extra in ["depth_vs_ror2_abs", "duration_over_period", "duration_over_aRs"]:
    if extra in canon.columns:
        feat_cols.append(extra)
feat_cols = sorted(set(feat_cols))
if not feat_cols:
    raise ValueError("No usable feature columns found after canonicalization. Check schema_map mappings.")

# ----------------------------
# 3) Map labels → integers 0/1/2
# ----------------------------
LAB2ID = {"FALSE POSITIVE": 0, "CANDIDATE": 1, "CONFIRMED": 2}

label_str = canon.get("label", pd.Series(index=canon.index, dtype=object)).astype(str).str.upper()
label_id = label_str.map(LAB2ID)

# Keep only rows with known labels
mask_labeled = label_id.notna()
canon = canon.loc[mask_labeled].copy()
label_id = label_id.loc[mask_labeled]

if canon.empty:
    raise ValueError("No labeled rows after canonicalization. Verify label fields in schema_map for your CSVs.")

X = canon[feat_cols].copy()
y = label_id.astype(int)

print(f"[data] Features: {len(feat_cols)}; Rows: {len(X):,}")
print("[data] Class balance:", dict(pd.Series(y).value_counts().sort_index().to_dict()))

# --------------------------------------------
# 4) Preprocess (impute + scale for linear part)
# --------------------------------------------
pre = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc", StandardScaler(with_mean=True, with_std=True))
])

# -------------------------------
# 5) Base learners for stacking
# -------------------------------
learners = []
if HAS_XGB:
    learners.append(("xgb", XGBClassifier(
        n_estimators=400, max_depth=6, subsample=0.7, colsample_bytree=0.8,
        learning_rate=0.05, reg_lambda=1.0, eval_metric="mlogloss",
        n_jobs=-1, tree_method="hist", random_state=42
    )))
learners.append(("rf", RandomForestClassifier(
    n_estimators=600, max_depth=None, min_samples_leaf=2, n_jobs=-1,
    class_weight="balanced_subsample", random_state=42
)))
learners.append(("lr", LogisticRegression(
    max_iter=2000, multi_class="multinomial", C=1.0
)))

stack = StackingClassifier(
    estimators=learners,
    final_estimator=LogisticRegression(max_iter=2000, multi_class="multinomial", C=1.0, random_state=42),
    passthrough=True,
    n_jobs=-1
)

# ---------------------------------------------
# 6) Train/val split → stack → probability cal
# ---------------------------------------------
# Need at least 2 classes to proceed
if len(np.unique(y)) < 2:
    raise ValueError(f"Training set has <2 classes (found {np.unique(y)}). You need at least two classes present.")

Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

pre.fit(Xtr, ytr)
Xtr_p = pre.transform(Xtr)
Xva_p = pre.transform(Xva)

stack.fit(Xtr_p, ytr)

# Isotonic may fail if each class isn’t present adequately in folds; fall back to sigmoid
try:
    cal = CalibratedClassifierCV(stack, method="isotonic", cv=5)
    cal.fit(Xva_p, yva)
except ValueError as e:
    print("[warn] Isotonic calibration failed:", e, "→ Falling back to sigmoid.")
    cal = CalibratedClassifierCV(stack, method="sigmoid", cv=5)
    cal.fit(Xva_p, yva)

# quick val accuracy
from sklearn.metrics import accuracy_score
pred_va = cal.predict(Xva_p)
acc = accuracy_score(yva, pred_va)
print(f"[val] Accuracy: {acc*100:.2f}% on {len(yva)} rows")

# ----------------
# 7) Save artifacts
# ----------------
joblib.dump(pre, OUT / "preproc.pkl")
joblib.dump(cal, OUT / "model.pkl")

meta = {
    "features": feat_cols,
    "labels": {0: "FALSE POSITIVE", 1: "CANDIDATE", 2: "CONFIRMED"},
    "source_mix": df_raw["__src"].value_counts().to_dict(),
    "note": "Stacked ensemble (XGB+RF+LR) with probability calibration (isotonic/sigmoid fallback).",
    "val_accuracy": acc
}
(OUT / "metadata.json").write_text(json.dumps(meta, indent=2))
print("Saved to:", OUT)
