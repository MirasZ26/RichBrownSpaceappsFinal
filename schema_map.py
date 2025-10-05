import pandas as pd
from typing import Dict, List, Optional

CANON_KEYS = [
    "period_days", "duration_hours", "depth_ppm", "ror", "a_over_rstar",
    "impact_b", "snr", "num_transits",
    "teff_K", "logg", "feh", "rstar_rsun", "mstar_msun",
    "insol_earth", "teq_K",
    "label"
]

MIN_REQUIRED = ["period_days", "duration_hours", "depth_ppm", "ror", "a_over_rstar", "snr"]

# All mapping keys below are LOWERCASE (we lowercase incoming headers first).
SOURCE_MAPS: Dict[str, Dict[str, List[str]]] = {
    "kepler": {
        "period_days":   ["koi_period"],
        "duration_hours":["koi_duration"],
        "depth_ppm":     ["koi_depth"],
        "ror":           ["koi_ror"],
        "a_over_rstar":  ["koi_dor"],
        "impact_b":      ["koi_impact"],
        "snr":           ["koi_model_snr", "koi_max_mult_ev"],
        "num_transits":  ["koi_num_transits"],
        "teff_K":        ["koi_steff"],
        "logg":          ["koi_slogg"],
        "feh":           ["koi_smet"],
        "rstar_rsun":    ["koi_srad"],
        "mstar_msun":    ["koi_smass"],
        "insol_earth":   ["koi_insol"],
        "teq_K":         ["koi_teq"],
        "label":         ["koi_disposition"],
    },
    "tess": {
        # common TOI synonyms; all lowercased
        "period_days":   ["toi_period", "orbital_period", "pl_orbper"],
        "duration_hours":["toi_duration", "transit_duration"],
        "depth_ppm":     ["toi_depth"],
        "ror":           ["toi_rp_rs", "rp_rs"],
        "a_over_rstar":  ["toi_a_rs", "a_rs"],
        "impact_b":      ["toi_impact", "impact_b"],
        "snr":           ["toi_snr", "mes"],
        "num_transits":  ["toi_n_transits"],
        "teff_K":        ["st_teff", "teff"],
        "logg":          ["st_logg", "logg"],
        "feh":           ["st_metfe", "feh"],
        "rstar_rsun":    ["st_rad", "rstar"],
        "mstar_msun":    ["st_mass", "mstar"],
        "insol_earth":   ["insol"],
        "teq_K":         ["teq"],
        "label":         ["tfopwg_disp", "disposition", "toi_disposition"],
    }
}

LABEL_MAPS = {
    "kepler": {
        "CONFIRMED": "CONFIRMED",
        "CANDIDATE": "CANDIDATE",
        "FALSE POSITIVE": "FALSE POSITIVE",
    },
    "tess": {
        "CP": "CONFIRMED",
        "CONFIRMED": "CONFIRMED",
        "PC": "CANDIDATE",
        "CANDIDATE": "CANDIDATE",
        "FP": "FALSE POSITIVE",
        "FALSE POSITIVE": "FALSE POSITIVE",
    }
}

def detect_source(df: pd.DataFrame) -> Optional[str]:
    cols = set(df.columns)
    # we lowercase headers before calling this
    if any(str(c).startswith("koi_") for c in cols):
        return "kepler"
    if any(k in cols for k in ["tfopwg_disp", "toi_id", "toi_period", "disposition"]):
        return "tess"
    return None

def _first_present(df: pd.DataFrame, candidates: List[str]):
    for c in candidates:
        if c in df.columns:
            return df[c]
    return None

def to_canonical(df: pd.DataFrame, source: Optional[str] = None) -> pd.DataFrame:
    # 🔧 Normalize headers first so detection/mapping is robust
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    if source is None:
        source = detect_source(df)
    if source is None:
        raise ValueError("Could not detect source. Provide `source` explicitly.")

    cmap = SOURCE_MAPS[source]
    out = {}
    for key, candidates in cmap.items():
        s = _first_present(df, candidates)
        if s is not None:
            out[key] = s

    canon = pd.DataFrame(out)

    # numeric coercion
    for col in canon.columns:
        if col != "label":
            canon[col] = pd.to_numeric(canon[col], errors="coerce")

    # standardize labels to unified strings (CONFIRMED/CANDIDATE/FALSE POSITIVE)
    if "label" in canon.columns:
        lmap = LABEL_MAPS.get(source, {})
        canon["label"] = (
            canon["label"].astype(str).str.upper().map(lambda x: lmap.get(x, x))
        )

    return canon
