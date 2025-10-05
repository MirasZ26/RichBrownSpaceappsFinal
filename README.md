# A World Away: Hunting for Exoplanets with AI (Kepler → TESS-ready)

End-to-end starter kit: tabular ML that classifies exoplanet candidates (CONFIRMED / CANDIDATE / FALSE POSITIVE) from NASA Kepler Cumulative first, with a clean path to add TESS TOI later via a canonical schema mapper.

## Quickstart

1) **Install deps** (recommend a fresh venv):
```bash
pip install -r requirements.txt
```

2) **Prepare data**  
   Export Kepler Cumulative CSV from the NASA Exoplanet Archive UI/API and save as `data/kepler_cumulative.csv`.

3) **Train**:
```bash
python train_ensemble.py --csv data/kepler_cumulative.csv --out_dir models/
```

4) **Run the app**:
```bash
streamlit run app_streamlit.py
```

Upload a CSV (Kepler or TESS when you map it) or enter values manually. The app shows predicted class probabilities, a label, and optional SHAP-based explanations.

## Files

- `schema_map.py` — Detects source (Kepler/TESS), maps raw columns → canonical schema, handles units.
- `data_prep.py` — Cleans, engineers features, splits, and returns training matrices.
- `train_kepler.py` — Trains XGBoost multiclass model, saves `model.pkl` + `metadata.json`.
- `app_streamlit.py` — Streamlit UI for training metadata display and predictions (CSV or manual form).
- `utils.py` — Shared helpers (metrics pretty-printing, model I/O, SHAP wrapper).

## Canonical Schema (minimal v1)

- `period_days` (float)  
- `duration_hours` (float)  
- `depth_ppm` (float)  
- `ror` (Rp/R*)  
- `a_over_rstar` (a/R*)  
- `impact_b` (float)  
- `snr` (signal-to-noise / MES-like)  
- `num_transits` (int)  
- `teff_K`, `logg`, `feh`, `rstar_rsun`, `mstar_msun` (stellar)  
- `insol_earth`, `teq_K` (irradiation/eq. temp)  
- `label` (CONFIRMED / CANDIDATE / FALSE POSITIVE)

## Notes

- The training script uses **XGBoost** with a sensible default config and stratified split.  
- You get physics-aware engineered features: depth vs ror² consistency, duration/period, duration/aR*.  
- Class imbalance handled via macro-F1 metric focus (and you can add class weights if desired).  
- To extend to **TESS**, add column names to `schema_map.py::SOURCE_MAPS['tess']` and ensure unit normalization.

## License
MIT
