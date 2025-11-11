# -*- coding: utf-8 -*-
from pathlib import Path
import joblib

def save_label_space(model, ckpt_dir: Path):
    try:
        classes_ = getattr(model.named_steps["clf"], "classes_", None)
        if classes_ is not None:
            (ckpt_dir / "meta.txt").write_text("classes:\n" + "\n".join(map(str, classes_)), encoding="utf-8")
    except Exception:
        pass

def load_pipeline(ckpt_dir: Path):
    return joblib.load(ckpt_dir / "best_pipeline.joblib")

def predict_texts(model, texts):
    return model.predict(texts)
