import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path("model/model.joblib")
METRICS_PATH = Path("model/metrics.json")

app = FastAPI(title="Temperature Prediction API", version="1.0.0")
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


class WeatherInput(BaseModel):
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity (%)")
    wind_speed: float = Field(..., ge=0, description="Wind speed (km/h)")
    meanpressure: float = Field(..., gt=900, lt=1100, description="Mean pressure (hPa)")
    month: int = Field(..., ge=1, le=12)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/metrics")
def metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(404, "No metrics found. Run train.py first.")
    return json.loads(METRICS_PATH.read_text())


@app.post("/predict")
def predict(x: WeatherInput):
    if model is None:
        raise HTTPException(503, "Model not trained. Run `python train.py` first.")
    row = pd.DataFrame([x.model_dump()])
    temp = float(model.predict(row)[0])
    return {"predicted_temperature_c": round(temp, 2)}
