# Temperature Prediction API

Shallow ANN (1 hidden layer, 32 ReLU units) that predicts mean daily temperature, served via **FastAPI**.

- **Stack:** Python, scikit-learn (`MLPRegressor`), FastAPI, Uvicorn
- **Dataset:** [Daily Delhi Climate (Kaggle)](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data) — place `DailyDelhiClimateTrain.csv` in `data/`
- **Features:** humidity, wind speed, mean pressure, month → **target:** mean temperature (°C)
- **Metrics:** see `model/metrics.json` after training (MAE / RMSE / R²) — <!-- paste numbers here -->

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py                      # trains + saves model/model.joblib
uvicorn app.main:app --reload        # http://127.0.0.1:8000/docs
```

## Example

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"humidity": 70, "wind_speed": 5, "meanpressure": 1008, "month": 6}'
# {"predicted_temperature_c": 31.4}
```

## Endpoints
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness + model loaded |
| GET | `/metrics` | test-set metrics |
| POST | `/predict` | predict temperature |
