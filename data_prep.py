import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from schema_map import to_canonical, MIN_REQUIRED

ENGINEERED = ["depth_vs_ror2_abs", "duration_over_period", "duration_over_aRs"]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-9

    if "depth_ppm" in df.columns and "ror" in df.columns:
        depth_frac = df["depth_ppm"] / 1e6
        df["depth_vs_ror2_abs"] = (depth_frac - (df["ror"] ** 2)).abs()
    else:
        df["depth_vs_ror2_abs"] = np.nan

    if "duration_hours" in df.columns and "period_days" in df.columns:
        duration_days = df["duration_hours"] / 24.0
        df["duration_over_period"] = duration_days / (df["period_days"] + eps)
    else:
        df["duration_over_period"] = np.nan

    if "duration_hours" in df.columns and "a_over_rstar" in df.columns:
        duration_days = df["duration_hours"] / 24.0
        df["duration_over_aRs"] = duration_days / (df["a_over_rstar"] + eps)
    else:
        df["duration_over_aRs"] = np.nan

    return df

def prepare_training_matrices(
    raw_df: pd.DataFrame,
    test_size: float = 0.15,
    random_state: int = 42,
    drop_missing_critical: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], Dict[str, float]]:
    df = to_canonical(raw_df, source=None)
    df = df[df["label"].isin(["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"])].copy()

    if drop_missing_critical:
        df = df.dropna(subset=[c for c in MIN_REQUIRED if c in df.columns])

    df = engineer_features(df)

    numeric_cols = [
        "period_days","duration_hours","depth_ppm","ror","a_over_rstar","impact_b",
        "snr","num_transits","teff_K","logg","feh","rstar_rsun","mstar_msun",
        "insol_earth","teq_K"
    ]
    present_numeric = [c for c in numeric_cols if c in df.columns]
    feat_cols = present_numeric + [c for c in ["depth_vs_ror2_abs","duration_over_period","duration_over_aRs"] if c in df.columns]

    X = df[feat_cols].astype(float)
    X = X.fillna(X.median(numeric_only=True))
    y = df["label"].astype("category")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    classes = np.unique(y_train)
    from sklearn.utils.class_weight import compute_class_weight
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = {cls: float(w) for cls, w in zip(classes, weights)}

    return X_train, X_test, y_train, y_test, feat_cols, class_weight_dict
