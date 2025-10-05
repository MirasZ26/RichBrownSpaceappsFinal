import json
import joblib
from pathlib import Path
from typing import Any, Dict

def save_model(model, out_dir: str, feature_names, label_encoder=None, metadata: Dict[str, Any]=None):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, f"{out_dir}/model.pkl")
    joblib.dump({"features": feature_names, "label_encoder": label_encoder}, f"{out_dir}/preproc.pkl")
    if metadata is not None:
        with open(f"{out_dir}/metadata.json","w") as f:
            json.dump(metadata, f, indent=2)

def load_model(model_dir: str):
    model = joblib.load(f"{model_dir}/model.pkl")
    pre = joblib.load(f"{model_dir}/preproc.pkl")
    meta = None
    p = Path(f"{model_dir}/metadata.json")
    if p.exists():
        meta = json.load(open(p))
    return model, pre, meta
