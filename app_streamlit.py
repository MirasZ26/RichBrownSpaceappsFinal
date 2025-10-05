import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timezone
from pandas.errors import EmptyDataError

from utils import load_model
from schema_map import to_canonical, CANON_KEYS
from data_prep import engineer_features
from habitability import make_plot

# ---------- App basics ----------
st.set_page_config(page_title="Terramicus — Exoplanet Hunter", layout="wide", page_icon="🪐")

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
DB_MAIN_PATH = (BASE_DIR / "data" / "kepler_cumulative.csv")
DB_USER_PATH = (BASE_DIR / "data" / "user_added.csv")

# brand/theme helpers
PRIMARY = "#FFE8A3"
NAVY = "#0A0E3F"
PURPLE = "#151A54"

def inject_css():
    st.markdown(f"""
    <style>
    .teramicus-nav {{
        background: rgba(21,26,84,0.65);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 10px 16px;
        margin-bottom: 10px;
        backdrop-filter: blur(8px);
    }}
    .brand {{
        font-weight: 700;
        font-size: 22px;
        letter-spacing: 0.5px;
        color: {PRIMARY};
    }}
    .hero {{
        background: linear-gradient(180deg, rgba(10,14,63,0.0) 0%, rgba(10,14,63,0.35) 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 32px;
        margin-top: 8px;
    }}
    .cta {{
        font-size: 18px !important;
        font-weight: 700;
        color: #111;
        background: {PRIMARY};
        padding: 12px 20px;
        border-radius: 14px;
        border: 0;
    }}
    .bigtitle {{ font-size: 54px; line-height: 1.06; font-weight: 800; }}
    .subtitle {{ font-size: 22px; opacity: 0.9; }}
    .pill {{
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 6px 10px; border-radius: 12px; display: inline-block;
    }}
    </style>
    """, unsafe_allow_html=True)

inject_css()

# ---------- Shared helpers (DB + similarity + labels) ----------
SHOW_COLS = [
    "period_days","duration_hours","depth_ppm","ror","a_over_rstar",
    "teff_K","rstar_rsun","mstar_msun","insol_earth","teq_K",
    "snr","num_transits","impact_b"
]

SIM_FEATURES = SHOW_COLS[:]  # reuse

LBL_MAP = {0: "FALSE POSITIVE", 1: "CANDIDATE", 2: "CONFIRMED"}
def to_text_label(x):
    try:
        i = int(x)
        return LBL_MAP.get(i, str(x)).upper()
    except Exception:
        s = str(x).upper()
        # normalize common forms
        repl = {"FP":"FALSE POSITIVE", "FALSE_POSITIVE":"FALSE POSITIVE"}
        return repl.get(s, s)

@st.cache_resource
def _load(model_dir: str):
    return load_model(model_dir)

def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except (FileNotFoundError, EmptyDataError):
        return pd.DataFrame()

def save_user_added(df: pd.DataFrame) -> None:
    DB_USER_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DB_USER_PATH, index=False)

def load_user_added(classes: list[str]) -> pd.DataFrame:
    DB_USER_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = _safe_read_csv(DB_USER_PATH)
    cols = ["name","db_origin","added_at","pred_label","db_class"] + list(CANON_KEYS) + [f"P({c})" for c in classes]
    if df.empty:
        df = pd.DataFrame(columns=cols)
        save_user_added(df)
    else:
        for col in cols:
            if col not in df.columns:
                df[col] = np.nan
    df["db_origin"] = df["db_origin"].fillna("manual").astype(str).str.strip().str.lower()
    df["db_class"] = df["db_class"].fillna("—").astype(str)
    return df

def load_kepler_main() -> pd.DataFrame:
    DB_MAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    return _safe_read_csv(DB_MAIN_PATH)

def _ensure_prob_cols(df: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    for c in classes:
        col = f"P({c})"
        if col not in df.columns:
            df[col] = np.nan
    return df

def _apply_show_order(df: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    ordered = ["name","db_class","pred_label","db_origin","added_at",
               *[f"P({c})" for c in classes], *SHOW_COLS]
    cols, seen = [], set()
    for c in ordered:
        if c in df.columns and c not in seen:
            cols.append(c); seen.add(c)
    for m in ["db_origin","added_at","db_class","pred_label"]:
        if m not in df.columns: df[m] = np.nan
    return df[cols] if cols else df

def kepler_to_display(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty:
        return pd.DataFrame(columns=["name","db_class","db_origin"])
    df = df_raw.copy()

    # name: kepler_name → kepoi_name → KIC ####
    names = pd.Series([None] * len(df), index=df.index, dtype="object")
    if "kepler_name" in df.columns:
        s = df["kepler_name"].astype("string")
        s = s.where(~s.isna() & (s.str.strip() != "") & (s.str.lower() != "nan"))
        names = names.combine_first(s)
    if "kepoi_name" in df.columns:
        s = df["kepoi_name"].astype("string")
        s = s.where(~s.isna() & (s.str.strip() != "") & (s.str.lower() != "nan"))
        names = names.combine_first(s)
    if "kepid" in df.columns:
        sid = pd.to_numeric(df["kepid"], errors="coerce").astype("Int64")
        s = sid.map(lambda x: f"KIC {int(x)}" if pd.notna(x) else None)
        names = names.combine_first(pd.Series(s, index=df.index, dtype="object"))
    df["name"] = names.fillna("—").astype(str)

    # classification → db_class (if present)
    set_class = False
    for col in ("koi_disposition", "koi_pdisposition", "disposition"):
        if col in df.columns:
            df["db_class"] = (
                df[col].astype(str).str.strip().str.upper()
                  .replace({"NAN": "—", "NONE": "—", "": "—"})
            )
            set_class = True
            break
    if not set_class:
        df["db_class"] = "—"

    # copy canonical display columns
    try:
        canon = to_canonical(df_raw, source=None)
        canon = engineer_features(canon)
        for c in SHOW_COLS:
            if c in canon.columns:
                df[c] = canon[c]
    except Exception:
        pass

    df["db_origin"] = "kepler"
    return df

def build_combined_db(classes: list[str]) -> pd.DataFrame:
    main = kepler_to_display(load_kepler_main())
    user = load_user_added(classes)
    admin = ["name","pred_label","db_origin","db_class","added_at"]
    for df in (main, user):
        for col in admin:
            if col not in df.columns:
                df[col] = np.nan
    main = _ensure_prob_cols(main, classes)
    user = _ensure_prob_cols(user, classes)
    for df in (main, user):
        df["db_origin"] = df["db_origin"].fillna("kepler").astype(str).str.strip().str.lower()
        df["db_class"] = df["db_class"].fillna("—").astype(str)
    all_cols = list(sorted(set(main.columns) | set(user.columns)))
    main = main.reindex(columns=all_cols)
    user = user.reindex(columns=all_cols)
    frames = [df for df in (main, user) if not df.empty and not df.isna().all(axis=None)]
    both = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame(columns=all_cols)
    return _apply_show_order(both, classes)

def find_similar(candidate_vals: dict, db: pd.DataFrame):
    if db is None or db.empty:
        return None, {"reason": "empty DB"}
    df = db.copy()
    if "db_origin" in df.columns:
        df = df[df["db_origin"].astype(str).str.lower() == "kepler"]
    if df.empty:
        return None, {"reason": "no Kepler rows"}
    cols = [c for c in SIM_FEATURES if c in df.columns and c in candidate_vals]
    if not cols:
        return None, {"reason": "no shared features"}
    df_num = df[cols].apply(pd.to_numeric, errors="coerce")
    vec = pd.Series({c: float(candidate_vals.get(c)) for c in cols})
    scales = {}
    for c in cols:
        s = df_num[c].dropna()
        if len(s) >= 5:
            q = s.quantile([0.05, 0.95]); scale = float(q.iloc[1] - q.iloc[0])
        else:
            scale = float(s.std()) if len(s) > 1 else 1.0
        if not scale or np.isnan(scale) or scale == 0.0: scale = 1.0
        scales[c] = scale
    best_idx, best_score, best_n = None, np.inf, 0
    for idx, row in df_num.iterrows():
        common = [c for c in cols if not pd.isna(vec[c]) and not pd.isna(row[c])]
        if len(common) < 4: continue
        diff = (row[common] - vec[common]) / pd.Series({c: scales[c] for c in common})
        score = float(np.sqrt(np.nansum(diff.values ** 2) / len(common)))
        if score < best_score:
            best_idx, best_score, best_n = idx, score, len(common)
    if best_idx is None:
        return None, {"reason": "no row with ≥4 overlapping features"}
    return df.loc[best_idx], {"score": best_score, "n": best_n}

# ---------- Top navbar ----------
def top_nav():
    with st.container():
        st.markdown('<div class="teramicus-nav">', unsafe_allow_html=True)
        cols = st.columns([2,2,1.2,1.4,1.6,3])
        with cols[0]:
            st.markdown('<div class="brand">Terramicus</div>', unsafe_allow_html=True)
        with cols[2]:
            st.markdown("<div class='pill'>Home</div>", unsafe_allow_html=True)
        with cols[3]:
            st.markdown("<div class='pill'>Prediction</div>", unsafe_allow_html=True)
        with cols[4]:
            st.markdown("<div class='pill'>DB view</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # actual page selector (clean radio, top, horizontal)
    if "active_page" not in st.session_state:
        st.session_state["active_page"] = "Home"
    page = st.radio("", ["Home", "Prediction", "DB view"],
                    index=["Home","Prediction","DB view"].index(st.session_state["active_page"]),
                    horizontal=True, label_visibility="collapsed", key="page_radio")
    st.session_state["active_page"] = page
    return page

# ---------- Pages ----------
def render_home():
    st.markdown("<br/>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("<div class='hero'>", unsafe_allow_html=True)
        st.markdown("<div class='bigtitle'>Searching for a new <span style='color:#FFE8A3'>home</span> for humanity.</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>Analyze Kepler/TESS candidates, visualize habitability, and grow your exoplanet database.</div>", unsafe_allow_html=True)
        if st.button("Start discovering!", help="Go to Prediction", use_container_width=False):
            st.session_state["active_page"] = "Prediction"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("### Contact")
        st.write("📧 zhansarin.m@nisa.edu.kz")
    with c2:
        img_path = ASSETS_DIR / "hero.jpg"
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
        else:
            st.info("Put a hero image at `assets/hero.jpg` to show it here.")

def render_db_view(classes_for_probs: list[str]):
    st.header("Exoplanet Database")
    combined = build_combined_db(classes_for_probs)
    st.session_state["combined_db"] = combined

    left, right = st.columns([2,2])
    origin_pick = left.selectbox("Origin filter", ["All", "Kepler DB", "Manual adds"], index=0, key="db_origin_pick")

    df_view = combined.copy()
    if "db_origin" not in df_view.columns:
        df_view["db_origin"] = "kepler"

    if origin_pick == "Kepler DB":
        df_view = df_view[df_view["db_origin"].astype(str).str.lower() == "kepler"]
    elif origin_pick == "Manual adds":
        df_view = df_view[df_view["db_origin"].astype(str).str.lower() == "manual"]

    if "db_class" in df_view.columns:
        first_idx = df_view.columns.get_loc("db_class")
        db_class_series = df_view.iloc[:, first_idx]
    else:
        db_class_series = pd.Series(dtype=str)

    cls_values = sorted([x for x in db_class_series.dropna().unique().tolist() if x != "—"])
    cls_pick = right.selectbox("Classification", ["Any"] + cls_values, index=0, key="db_class_pick")
    if cls_pick != "Any":
        df_view = df_view[db_class_series == cls_pick]

    st.dataframe(df_view, use_container_width=True, height=520)
    st.caption("Main DB: data/kepler_cumulative.csv   •   Manual adds: data/user_added.csv")

def render_prediction():
    st.header("Prediction")
    col1, col2 = st.columns([2,1])
    with col2:
        model_dir = st.text_input("Model directory", value="models",
                                  help="Folder with model.pkl, preproc.pkl, metadata.json")
        if st.button("Load model"):
            try:
                model, pre, meta = _load(model_dir)
                st.session_state["model_loaded"] = True
                st.session_state["model_dir"] = model_dir
                st.session_state["classes"] = list(model.classes_)
                st.success("Model loaded!")
            except Exception as e:
                st.error(f"Failed to load model: {e}")

    if not st.session_state.get("model_loaded"):
        st.info("Load a model to start predicting.")
        return

    model, pre, meta = _load(st.session_state["model_dir"])

    tabs = st.tabs(["CSV Upload", "Manual Entry"])

    # ---------- CSV Upload ----------
    with tabs[0]:
        st.write("Upload Kepler (or mapped TESS) CSV. We'll auto-detect source and map to canonical schema.")
        upl = st.file_uploader("CSV file", type=["csv"])
        if upl is not None:
            raw = pd.read_csv(upl)
            try:
                canon = to_canonical(raw, source=None)
                canon = engineer_features(canon)

                feat_names = getattr(getattr(model, "get_booster", lambda: None)(), "feature_names", None)
                if not feat_names:
                    feat_names = meta.get("features") if meta else pre.get("features")

                X = canon[[c for c in feat_names if c in canon.columns]].copy()
                X = X.fillna(X.median(numeric_only=True))

                # Predict all rows (2D proba)
                proba = model.predict_proba(X)  # shape: (n_rows, n_classes)
                preds_raw = np.array(model.classes_)[np.argmax(proba, axis=1)]
                preds_txt = [to_text_label(p) for p in preds_raw]

                out = canon.copy()
                for i, cls in enumerate(model.classes_):
                    out[f"P({to_text_label(cls)})"] = proba[:, i]
                out["pred_label"] = preds_txt
                out["Confidence (%)"] = (np.max(proba, axis=1) * 100).round(1)

                # If labels exist in the file, show accuracy; otherwise show avg confidence
                acc = None
                gt = None
                for col in ("koi_disposition", "koi_pdisposition", "disposition", "db_class", "pred_label"):
                    if col in raw.columns:
                        gt = raw[col].astype(str).str.upper()
                        break
                    if col in canon.columns:
                        gt = canon[col].astype(str).str.upper()
                        break
                if gt is not None:
                    gt = gt.replace({"FP": "FALSE POSITIVE", "FALSE_POSITIVE": "FALSE POSITIVE"})
                    try:
                        acc = float((gt.values == np.array(preds_txt)).mean())
                    except Exception:
                        acc = None

                st.write("### Predictions")
                m1, m2 = st.columns(2)
                m1.metric("Rows", len(out))
                if acc is not None and np.isfinite(acc):
                    m2.metric("Accuracy on uploaded labels", f"{acc*100:.1f}%")
                else:
                    m2.metric("Avg. confidence", f"{out['Confidence (%)'].mean():.1f}%")

                st.dataframe(out.head(50), use_container_width=True)



                # ---------- Habitability expander (single + multi overlay) ----------
                with st.expander("Habitability position (relative to Earth)", expanded=False):
                    if len(canon) > 0:
                        row_idx = st.number_input(
                            "Row to visualize",
                            min_value=0, max_value=len(canon)-1, value=0, step=1, key="hz_csv_row"
                        )
                        row = canon.iloc[int(row_idx)]
                        mode_csv = st.radio(
                            "Source for values",
                            ["Use dataset (star+orbit preferred)", "Manual override"],
                            horizontal=True, key="hz_csv_mode"
                        )
                        albedo_csv = st.number_input(
                            "Bond albedo A", value=0.30, min_value=0.0, max_value=0.95, step=0.05, key="hz_csv_albedo"
                        )

                        if mode_csv == "Use dataset (star+orbit preferred)":
                            kwargs = dict(
                                name=f"Row {int(row_idx)}",
                                S=float(row["insol_earth"]) if "insol_earth" in row and pd.notnull(row["insol_earth"]) else None,
                                star_teff_k=float(row["teff_K"]) if "teff_K" in row and pd.notnull(row["teff_K"]) else None,
                                star_radius_rsun=float(row["rstar_rsun"]) if "rstar_rsun" in row and pd.notnull(row["rstar_rsun"]) else None,
                                a_over_rstar=float(row["a_over_rstar"]) if "a_over_rstar" in row and pd.notnull(row["a_over_rstar"]) else None,
                                period_days=float(row["period_days"]) if "period_days" in row and pd.notnull(row["period_days"]) else None,
                                mstar_msun=float(row["mstar_msun"]) if "mstar_msun" in row and pd.notnull(row["mstar_msun"]) else None,
                                albedo=albedo_csv
                            )
                            fig, info = make_plot(**kwargs)
                            st.pyplot(fig, clear_figure=True)
                            st.caption(f"S = {info['S']:.2f} ⊕ · T_eq ≈ {info['Teq']:.0f} K (no greenhouse).")
                        else:
                            c1, c2 = st.columns(2)
                            pick = c1.selectbox(
                                "Provide", ["Star+Orbit", "Direct S (starlight rel. Earth)"],
                                index=0, key="hz_csv_pick"
                            )
                            if pick == "Star+Orbit":
                                teff_i  = c1.number_input("Host-star Teff (K)", value=5600, step=50, key="hz_csv_teff")
                                rstar_i = c1.number_input("Host-star radius (Rsun)", value=1.00, step=0.01, format="%.2f", key="hz_csv_rstar")
                                a_over_r_i = c2.number_input("a / R*", value=10.0, step=0.1, key="hz_csv_aoverr")
                                period_i   = c2.number_input("Orbital period (days)", value=10.0, step=0.1, key="hz_csv_period")
                                mstar_i    = c2.number_input("Stellar mass (M☉)", value=1.00, step=0.01, key="hz_csv_mstar")
                                fig, info = make_plot(
                                    name=f"Row {int(row_idx)}",
                                    star_teff_k=teff_i, star_radius_rsun=rstar_i,
                                    a_over_rstar=a_over_r_i, period_days=period_i, mstar_msun=mstar_i,
                                    albedo=albedo_csv
                                )
                                st.pyplot(fig, clear_figure=True)
                                st.caption(f"S = {info['S']:.2f} ⊕ · T_eq ≈ {info['Teq']:.0f} K (no greenhouse).")
                            else:
                                S_i = c1.number_input("S (starlight rel. Earth)", value=1.00, step=0.05, format="%.2f", key="hz_csv_S")
                                fig, info = make_plot(name=f"Row {int(row_idx)}", S=S_i, albedo=albedo_csv)
                                st.pyplot(fig, clear_figure=True)
                                st.caption(f"S = {info['S']:.2f} ⊕ · T_eq ≈ {info['Teq']:.0f} K (no greenhouse).")

                        # Multi-point overlay (with non-nesting legend control)
                        st.markdown("—")
                        st.markdown("**Plot multiple rows on the same habitability diagram**")
                        overlay_idxs = st.multiselect(
                            "Rows to overlay",
                            options=list(range(len(canon))),
                            default=list(range(min(5, len(canon)))),
                            help="Select multiple rows to plot together (first chosen row sets the background).",
                            key="hz_csv_multi_idxs"
                        )
                        if overlay_idxs:
                            T_SUN = 5772.0; RSUN_AU = 0.00465047
                            def _S_Teq_from_row(r: pd.Series, albedo: float):
                                if "insol_earth" in r and pd.notnull(r["insol_earth"]):
                                    S = float(r["insol_earth"])
                                else:
                                    teff = float(r["teff_K"]) if "teff_K" in r and pd.notnull(r["teff_K"]) else None
                                    rstar = float(r["rstar_rsun"]) if "rstar_rsun" in r and pd.notnull(r["rstar_rsun"]) else None
                                    dor = float(r["a_over_rstar"]) if "a_over_rstar" in r and pd.notnull(r["a_over_rstar"]) else None
                                    if teff is None or rstar is None or dor is None: return np.nan, np.nan
                                    Lrel = (rstar**2) * (teff / T_SUN)**4
                                    a_au = dor * rstar * RSUN_AU
                                    if a_au <= 0: return np.nan, np.nan
                                    S = Lrel / (a_au**2)
                                Teq = 278.5 * ((1.0 - albedo) ** 0.25) * (S ** 0.25) if S > 0 else np.nan
                                return S, Teq

                            r0 = canon.iloc[overlay_idxs[0]]
                            kwargs0 = dict(
                                name=f"Row {int(overlay_idxs[0])}",
                                S=float(r0["insol_earth"]) if "insol_earth" in r0 and pd.notnull(r0["insol_earth"]) else None,
                                star_teff_k=float(r0["teff_K"]) if "teff_K" in r0 and pd.notnull(r0["teff_K"]) else None,
                                star_radius_rsun=float(r0["rstar_rsun"]) if "rstar_rsun" in r0 and pd.notnull(r0["rstar_rsun"]) else None,
                                a_over_rstar=float(r0["a_over_rstar"]) if "a_over_rstar" in r0 and pd.notnull(r0["a_over_rstar"]) else None,
                                period_days=float(r0["period_days"]) if "period_days" in r0 and pd.notnull(r0["period_days"]) else None,
                                mstar_msun=float(r0["mstar_msun"]) if "mstar_msun" in r0 and pd.notnull(r0["mstar_msun"]) else None,
                                albedo=albedo_csv
                            )
                            fig, info = make_plot(**kwargs0); ax = fig.axes[0]

                            palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0","C1","C2","C3","C4","C5"])
                            overlay_labels = []
                            for j, idx in enumerate(overlay_idxs):
                                r = canon.iloc[idx]
                                S, Teq = _S_Teq_from_row(r, albedo_csv)
                                if not np.isfinite(S) or not np.isfinite(Teq): continue
                                if "kepler_name" in raw.columns and pd.notnull(raw.iloc[idx].get("kepler_name")) and str(raw.iloc[idx]["kepler_name"]).strip() != "":
                                    label = str(raw.iloc[idx]["kepler_name"])
                                elif "kepoi_name" in raw.columns and pd.notnull(raw.iloc[idx].get("kepoi_name")) and str(raw.iloc[idx]["kepoi_name"]).strip() != "":
                                    label = str(raw.iloc[idx]["kepoi_name"])
                                elif "kepid" in raw.columns and pd.notnull(raw.iloc[idx].get("kepid")):
                                    try: label = f"KIC {int(raw.iloc[idx]['kepid'])}"
                                    except Exception: label = f"Row {idx}"
                                else:
                                    label = f"Row {idx}"
                                color = palette[j % len(palette)]
                                ax.scatter(S, Teq, s=80, label=label, alpha=0.95, zorder=5, color=color)
                                overlay_labels.append((label, color))

                            # Legend control (no nested expanders)
                            legend_mode = st.radio(
                                "Legend display",
                                ["Hidden", "On plot (outside)", "Collapsible list below"],
                                index=2, horizontal=True, key="hz_csv_legend_mode"
                            )
                            if legend_mode == "On plot (outside)":
                                handles, labels = ax.get_legend_handles_labels()
                                if ax.get_legend(): ax.get_legend().remove()
                                fig.subplots_adjust(right=0.78)
                                ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5),
                                          fontsize=8, framealpha=0.9, ncols=1)
                                st.pyplot(fig, clear_figure=True)
                            elif legend_mode == "Hidden":
                                if ax.get_legend(): ax.get_legend().remove()
                                st.pyplot(fig, clear_figure=True)
                            else:
                                if ax.get_legend(): ax.get_legend().remove()
                                st.pyplot(fig, clear_figure=True)
                                show_list = st.checkbox("Show legend list (overlay points)", value=False, key="hz_csv_show_legend")
                                if show_list:
                                    st.caption("Hatched = too cold/hot; teal = conservative HZ; blue = optimistic HZ; Earth at S=1, T_eq≈255 K.")
                                    for label, color in overlay_labels:
                                        st.markdown(f"<span style='color:{color}'>●</span> {label}", unsafe_allow_html=True)

                # ---------- SHAP ----------
                with st.expander("Explain first 3 predictions (SHAP)", expanded=False):
                    try:
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X.head(3))
                        for i in range(min(3, len(X))):
                            st.write(f"#### Row {i}")
                            st.pyplot(shap.waterfall_plot(shap.Explanation(
                                values=shap_values[np.argmax(proba[i])][i],
                                base_values=explainer.expected_value[np.argmax(proba[i])],
                                data=X.head(3).iloc[i].values,
                                feature_names=X.columns.tolist()
                            )))
                    except Exception as e:
                        st.info(f"SHAP explanation not shown: {e}")
            except Exception as e:
                st.error(f"Failed to map/predict: {e}")

    # ---------- Manual Entry ----------
    with tabs[1]:
        st.write("Enter values manually (canonical schema). Missing values default to NaN and are imputed when possible.")
        defaults = {
            "period_days": 10.0, "duration_hours": 2.5, "depth_ppm": 500.0, "ror": 0.05,
            "a_over_rstar": 10.0, "impact_b": 0.5, "snr": 12.0, "num_transits": 3,
            "teff_K": 5750.0, "logg": 4.4, "feh": 0.0, "rstar_rsun": 1.0,
            "mstar_msun": 1.0, "insol_earth": 1.0, "teq_K": 280.0,
        }
        vals = {}
        cols = st.columns(3)
        keys = [k for k in CANON_KEYS if k != "label"]
        for i, key in enumerate(keys):
            with cols[i % 3]:
                if key in ["num_transits"]:
                    vals[key] = st.number_input(key, value=int(defaults.get(key, 0)), step=1)
                else:
                    vals[key] = st.number_input(key, value=float(defaults.get(key, 0.0)))

        if st.button("Predict (manual)"):
            one = pd.DataFrame([vals]); one = engineer_features(one)
            feat_names = getattr(getattr(model, "get_booster", lambda: None)(), "feature_names", None)
            if not feat_names: feat_names = meta.get("features") if meta else pre.get("features")
            X = one[[c for c in feat_names if c in one.columns]].copy()
            X = X.fillna(X.median(numeric_only=True))
            proba = model.predict_proba(X)[0]          # 1D vector for the single row
            pred_idx = int(np.argmax(proba))
            pred_class = model.classes_[pred_idx]
            pred = to_text_label(pred_class)
            conf = float(proba[pred_idx])

            conf = float(np.max(proba))
            m1, m2 = st.columns(2)
            m1.metric("Predicted label", pred)
            m2.metric("Confidence", f"{conf*100:.1f}%")

            st.write({to_text_label(cls): float(p) for cls, p in zip(model.classes_, proba)})

            # save for add-to-DB
            st.session_state["manual_last"] = {
                "vals": vals.copy(),
                "proba": proba.tolist(),
                "pred": pred,  # text label
                "classes": list(model.classes_),
                "stamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }


            # closest match in Kepler
            try:
                combined_now = st.session_state.get("combined_db", build_combined_db(list(model.classes_)))
                match, info = find_similar(vals, combined_now)
                with st.expander("Closest match in Kepler DB", expanded=False):
                    if match is None:
                        st.info(f"No similar planet found ({info.get('reason')}).")
                    else:
                        st.write(
                            f"Most similar: **{match.get('name','?')}** · class: `{match.get('db_class','—')}` "
                            f"· overlapping features: {info['n']} · score: {info['score']:.2f}"
                        )
                        show_cols = ["name","db_class"] + [c for c in SIM_FEATURES if c in match.index]
                        st.dataframe(match[show_cols].to_frame().T, use_container_width=True)
            except Exception as e:
                st.info(f"Similarity search not available: {e}")

            # Habitability (manual)
            with st.expander("Habitability position (relative to Earth)", expanded=True):
                c1, c2 = st.columns(2)
                mode = c1.selectbox("Provide", ["Use inputs above", "Star+Orbit (override)", "Direct S (override)"],
                                    index=0, key="hz_man_mode")
                albedo = c2.number_input("Bond albedo A", value=0.30, min_value=0.0, max_value=0.95, step=0.05, key="hz_man_albedo")
                if mode == "Use inputs above":
                    try:
                        fig, info = make_plot(
                            name="Manual candidate",
                            S=vals.get("insol_earth"), star_teff_k=vals.get("teff_K"),
                            star_radius_rsun=vals.get("rstar_rsun"), a_over_rstar=vals.get("a_over_rstar"),
                            period_days=vals.get("period_days"), mstar_msun=vals.get("mstar_msun"), albedo=albedo
                        )
                        st.pyplot(fig, clear_figure=True); st.caption(f"S = {info['S']:.2f} ⊕ · T_eq ≈ {info['Teq']:.0f} K")
                    except Exception as e:
                        st.info(f"Unable to plot from inputs above: {e}")
                elif mode == "Star+Orbit (override)":
                    teff_i  = c1.number_input("Host-star Teff (K)", value=float(vals.get("teff_K", 5600.0)), step=50, key="hz_man_teff")
                    rstar_i = c1.number_input("Host-star radius (Rsun)", value=float(vals.get("rstar_rsun", 1.0)), step=0.01, format="%.2f", key="hz_man_rstar")
                    a_over_r_i = c2.number_input("a / R*", value=float(vals.get("a_over_rstar", 10.0)), step=0.1, key="hz_man_aoverr")
                    period_i   = c2.number_input("Orbital period (days)", value=float(vals.get("period_days", 10.0)), step=0.1, key="hz_man_period")
                    mstar_i    = c2.number_input("Stellar mass (M☉)", value=float(vals.get("mstar_msun", 1.0)), step=0.01, key="hz_man_mstar")
                    fig, info = make_plot(name="Manual candidate", star_teff_k=teff_i, star_radius_rsun=rstar_i,
                                          a_over_rstar=a_over_r_i, period_days=period_i, mstar_msun=mstar_i, albedo=albedo)
                    st.pyplot(fig, clear_figure=True); st.caption(f"S = {info['S']:.2f} ⊕ · T_eq ≈ {info['Teq']:.0f} K")
                else:
                    S_i = c1.number_input("S (starlight rel. Earth)", value=float(vals.get("insol_earth", 1.0)),
                                          step=0.05, format="%.2f", key="hz_man_S")
                    fig, info = make_plot(name="Manual candidate", S=S_i, albedo=albedo)
                    st.pyplot(fig, clear_figure=True); st.caption(f"S = {info['S']:.2f} ⊕ · T_eq ≈ {info['Teq']:.0f} K")

        # add-to-DB (persists)
        last = st.session_state.get("manual_last")
        if last:
            st.divider()
            st.write("**Add this prediction to the database**")
            default_name = f"Manual-{last['stamp'].replace(':','').replace('-','')}"
            new_name = st.text_input("Name this exoplanet", value=default_name, key="db_add_name")
            if st.button("Add to database", key="db_add_btn"):
                row = {**last["vals"]}
                row.update({f"P({to_text_label(c)})": float(p) for c, p in zip(last["classes"], last["proba"])})
                row["name"] = new_name
                row["pred_label"] = str(last["pred"])
                row["db_origin"] = "manual"
                row["db_class"] = "—"
                row["added_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                user_df = load_user_added(last["classes"])
                for c in CANON_KEYS:
                    if c not in user_df.columns: user_df[c] = np.nan
                user_df = pd.concat([user_df, pd.DataFrame([row])], ignore_index=True, sort=False)
                save_user_added(user_df)
                st.session_state["combined_db"] = build_combined_db(last["classes"])
                st.success(f"Added “{new_name}” to manual database ({DB_USER_PATH}).")
                st.rerun()

# ---------- Router ----------
page = top_nav()
if page == "Home":
    render_home()
elif page == "Prediction":
    render_prediction()
else:
    # if a model is loaded, use its classes for P() columns; else none
    classes = st.session_state.get("classes", [])
    render_db_view(classes)
