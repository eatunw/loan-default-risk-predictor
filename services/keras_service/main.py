from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import os
import numpy as np
from typing import Dict

app = FastAPI(title="Keras Model Service")


class PredictRequest(BaseModel):
    features: Dict[str, float]


MODEL = None
SCALER = None
FEATURE_NAMES = []


def load_artifacts():
    global MODEL, SCALER, FEATURE_NAMES
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    feature_path = os.path.join(base, "feature_names.json")
    scaler_path = os.path.join(base, "scaler.pkl")
    model_path = os.path.join(base, "loan_default_model_v1.keras")

    if os.path.exists(feature_path):
        with open(feature_path, "r") as f:
            FEATURE_NAMES = json.load(f)

    try:
        from joblib import load as _load
        if os.path.exists(scaler_path):
            SCALER = _load(scaler_path)
    except Exception:
        SCALER = None

    try:
        # import TensorFlow lazily
        from tensorflow.keras.models import load_model
        if os.path.exists(model_path):
            MODEL = load_model(model_path)
    except Exception as e:
        MODEL = None
        print("Warning: failed to load Keras model:", e)


@app.on_event("startup")
def startup_event():
    load_artifacts()


@app.get("/health")
def health():
    return {"status": "ok", "keras_loaded": MODEL is not None}


@app.post("/predict")
def predict(req: PredictRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Keras model not available")

    # Build feature vector in correct order
    if FEATURE_NAMES:
        x = [req.features.get(name, 0.0) for name in FEATURE_NAMES]
    else:
        keys = sorted(req.features.keys())
        x = [req.features[k] for k in keys]

    arr = np.array(x, dtype=float).reshape(1, -1)
    if SCALER is not None:
        try:
            arr = SCALER.transform(arr)
        except Exception:
            pass

    try:
        pred = MODEL.predict(arr)
        prob = float(pred.ravel()[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"prediction error: {e}")

    return {"probability": prob}
