# Temperature Prediction API

A shallow neural network (ANN) that predicts mean daily temperature from weather conditions, deployed as a REST API with **FastAPI**.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-009688) ![scikit--learn](https://img.shields.io/badge/scikit--learn-ANN-orange)

## Overview
| | |
|---|---|
| **Task** | Regression: predict mean temperature (°C) |
| **Model** | ANN with 1 hidden layer (32 ReLU units), standardized inputs |
| **Dataset** | [Daily Delhi Climate](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data) (Kaggle) |
| **Features** | humidity, wind speed, mean pressure, month |
| **Serving** | FastAPI + Uvicorn, with auto-generated Swagger docs |

## Results
Evaluated on a held-out 20% test split:

| MAE (°C) | RMSE (°C) | R² |
|---|---|---|
| **X.XX** | **X.XX** | **0.XX** |

<!-- Replace X's with the values from model/metrics.json -->

## Project structure
```
├── app/
│   └── main.py        # FastAPI app (/predict, /health, /metrics)
├── data/              # put DailyDelhiClimateTrain.csv here
├── model/             # trained model + metrics.json (generated)
├── train.py           # data loading, training, evaluation
└── requirements.txt
```

## Getting started
```bash
git clone https://github.com/sathvik70004-cmyk/temperature-prediction-api.git
cd temperature-prediction-api

python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

# download DailyDelhiClimateTrain.csv from Kaggle into data/
python train.py
python -m uvicorn app.main:app --reload
```
Open **http://127.0.0.1:8000/docs** for the interactive API docs.

## API
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Predict temperature from weather inputs |
| `GET` | `/metrics` | Test-set MAE, RMSE, R² |
| `GET` | `/health` | Service and model status |

**Request**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"humidity": 70, "wind_speed": 5, "meanpressure": 1008, "month": 6}'
```
**Response**
```json
{"predicted_temperature_c": 31.4}
```
Inputs are validated with Pydantic (e.g. humidity 0–100, month 1–12); invalid requests return a `422` error.

## Screenshot
![Swagger UI](docs/swagger.png)
<!-- Take a screenshot of /docs, save it as docs/swagger.png -->

## Future improvements
- Add lag features / time-series model (LSTM) for forecasting
- Dockerize and deploy (Render / Railway)
- Add unit tests and CI with GitHub Actions
