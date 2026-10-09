"""Train a shallow ANN (1 hidden layer) to predict mean daily temperature.

Dataset: Daily Delhi Climate (Kaggle) -> data/DailyDelhiClimateTrain.csv
Columns: date, meantemp, humidity, wind_speed, meanpressure
If the CSV is missing, a synthetic dataset with the same schema is used so the
pipeline runs end to end.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["humidity", "wind_speed", "meanpressure", "month"]
TARGET = "meantemp"


def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.exists():
        df = pd.read_csv(p, parse_dates=["date"])
        df["month"] = df["date"].dt.month
        # drop obvious sensor outliers in pressure
        df = df[(df["meanpressure"] > 900) & (df["meanpressure"] < 1100)]
        print(f"Loaded real dataset: {len(df)} rows from {p}")
        return df.dropna(subset=FEATURES + [TARGET])

    print("CSV not found -> generating synthetic data (replace with real dataset).")
    rng = np.random.default_rng(42)
    n = 1500
    month = rng.integers(1, 13, n)
    seasonal = 25 + 8 * np.sin((month - 4) / 12 * 2 * np.pi)
    humidity = np.clip(80 - (seasonal - 17) * 2 + rng.normal(0, 8, n), 10, 100)
    wind = np.abs(rng.normal(6, 3, n))
    pressure = rng.normal(1008, 6, n)
    temp = seasonal - 0.12 * (humidity - 60) - 0.1 * wind - 0.2 * (pressure - 1008) + rng.normal(0, 1.2, n)
    return pd.DataFrame(
        {"meantemp": temp, "humidity": humidity, "wind_speed": wind, "meanpressure": pressure, "month": month}
    )


def main(data_path: str) -> None:
    df = load_data(data_path)
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "ann",
                MLPRegressor(
                    hidden_layer_sizes=(32,),  # shallow: single hidden layer
                    activation="relu",
                    max_iter=800,
                    early_stopping=True,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    metrics = {
        "mae": round(float(mean_absolute_error(y_te, pred)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_te, pred))), 3),
        "r2": round(float(r2_score(y_te, pred)), 3),
        "n_train": len(X_tr),
        "n_test": len(X_te),
    }
    print("Test metrics:", metrics)

    Path("model").mkdir(exist_ok=True)
    joblib.dump(model, "model/model.joblib")
    Path("model/metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/DailyDelhiClimateTrain.csv")
    main(ap.parse_args().data)
