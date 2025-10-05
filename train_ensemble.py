# train_ensemble.py — Kepler-only stacked ensemble (saves to models/)
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from pandas.errors import ParserError

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Optional XGBoost as one of the base learners
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# Project imports
from schema_map import to_canonical, CANON_KEYS
from data_prep import engineer_features

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "kepler_cumulative.csv"
OUT  = BASE / "models"                      # <— save directly in models/
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Robust CSV reader (handles minor quirks gracefully)
# ----------------------------------------------------------------------
def _safe_read(path: Path, **kwargs) -> pd.DataFrame:
    """Call pd.read_csv with version-safe arguments."""
    try:
        return pd.read_csv(path, **kwargs)
    except TypeError:
        # Drop args that older pandas don’t understand
        kwargs2 = dict(kwargs)
        kwargs2.pop("on_bad_lines", None)
        if kwargs2.get("engine") == "python":
            kwargs2.pop("low_memory", None)
        # Legacy flags for very old pandas (harmless otherwise)
        kwargs2.setdefault("error_bad_lines", False)
        kwargs2.setdefault("warn_bad_lines", False)
        return pd.read_csv(path, **kwargs2)

def smart_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    # 1) fast path
    try:
        df = _safe_read(path, low_memory=False, on_bad_lines="skip")
    except ParserError:
        # 2) python engine with sniffed delimiter
        try:
            df = _safe_read(
                path, sep=None, engine="python",
                on_bad_lines="skip", comment="#", skip_blank_lines=True,
                encoding_errors="ignore"
            )
        except (ParserError, ValueError):
            # 3) tab fallback
            try:
                df = _safe_read(
                    path, sep="\t", engine="python",
                    on_bad_lines="skip", comment="#", skip_blank_lines=True,
                    encoding_errors="ignore"
                )
            except Exception:
                # 4) semicolon fallback
                df = _safe_read(
                    path, sep=";", engine="python",
                    on_bad_lines="skip", comment="#", skip_blank_lines=True,
                    encoding_errors="ignore"
                )
    # Normalize headers (lowercase) for mapping
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

# ----------------------------------------------------------------------
# 1) Load Kepler ONLY
# ----------------------------------------------------------------------
print(f"[load] Kepler CSV: {DATA}")
df_raw = smart_read_csv(DATA)
if df_raw.empty:
    raise SystemExit("Kepler CSV is empty or unreadable. Put it at data/kepler_cumulative.csv")

# ----------------------------------------------------------------------
# 2) Canonicalize + engineer features
# ----------------------------------------------------------------------
# Force kepler mapping so TESS is never used
canon = to_canonical(df_raw, source="kepler")
canon = engineer_features(canon)

# ----------------------------------------------------------------------
# 3) Feature / label selection
# ----------------------------------------------------------------------
feat_cols = [c for c in canon.columns if c in CANON_KEYS and c != "label"]
for extra in ["depth_vs_ror2_abs", "duration_over_period", "duration_over_aRs", "duration_over_ars"]:
    if extra in canon.columns and extra not in feat_cols:
        feat_cols.append(extra)

# labels → integers {FP:0, CANDIDATE:1, CONFIRMED:2}
if "label" not in canon.columns or canon["label"].isna().all():
    raise SystemExit("No labels found in Kepler data after canonicalization.")

canon = canon.dropna(subset=["label"]).copy()
if canon["label"].dtype == object or str(canon["label"].dtype).startswith("string"):
    LBL_TO_INT = {"FALSE POSITIVE": 0, "CANDIDATE": 1, "CONFIRMED": 2,
                  "FP": 0, "PC": 1, "CP": 2}
    canon["label"] = canon["label"].astype(str).str.upper().map(LBL_TO_INT)

if canon["label"].isna().any():
    bad = int(canon["label"].isna().sum())
    raise SystemExit(f"{bad} rows have unmapped labels. Clean/standardize koi_disposition first.")

X = canon[feat_cols].copy()
y = canon["label"].astype(int)

# ----------------------------------------------------------------------
# 4) Preprocess + base learners (diverse types)
# ----------------------------------------------------------------------
pre = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc",  StandardScaler(with_mean=True, with_std=True))
])

learners = []

if HAS_XGB:
    learners.append(("xgb", XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=500,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        learning_rate=0.05,
        reg_lambda=1.0,
        n_jobs=-1,
        tree_method="hist",
        random_state=42
    )))

learners.append(("rf", RandomForestClassifier(
    n_estimators=800,
    max_depth=None,
    min_samples_leaf=2,
    n_jobs=-1,
    class_weight="balanced_subsample",
    random_state=42
)))

learners.append(("gb", GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)))

meta_lr = LogisticRegression(max_iter=3000, multi_class="multinomial")

stack = StackingClassifier(
    estimators=learners,
    final_estimator=meta_lr,
    passthrough=True,
    n_jobs=-1
)

# ----------------------------------------------------------------------
# 5) Train/validate with calibration
# ----------------------------------------------------------------------
Xtr, Xva, ytr, yva = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pre.fit(Xtr, ytr)
Xtr_p = pre.transform(Xtr)
Xva_p = pre.transform(Xva)

stack.fit(Xtr_p, ytr)

cal = CalibratedClassifierCV(stack, method="isotonic", cv=5)
cal.fit(Xva_p, yva)

yhat = cal.predict(Xva_p)
acc = accuracy_score(yva, yhat)
print(f"[holdout] accuracy = {acc*100:.2f}%")
print(classification_report(yva, yhat, digits=3))

# ----------------------------------------------------------------------
# 6) Save artifacts for the Streamlit app (direct to models/)
# ----------------------------------------------------------------------
joblib.dump(pre, OUT / "preproc.pkl")
joblib.dump(cal, OUT / "model.pkl")

meta = {
    "features": feat_cols,
    "labels": {0: "FALSE POSITIVE", 1: "CANDIDATE", 2: "CONFIRMED"},
    "source_mix": {"kepler": int(len(canon))},
    "note": "Kepler-only stacked ensemble (XGB+RF+GB) with isotonic calibration."
}
(OUT / "metadata.json").write_text(json.dumps(meta, indent=2))

print("Saved artifacts to:", OUT.resolve())
print("➡ In the app, set Model directory to:", OUT)
