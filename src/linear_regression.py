from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_preprocess import load_house_splits
from utils import (
    append_metrics,
    default_model_path,
    ensure_dir,
    find_latest_model,
    save_prediction_scatter,
)


def _build_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MSE": f"{mse:.4f}",
        "RMSE": f"{mse ** 0.5:.4f}",
        "MAE": f"{mean_absolute_error(y_true, y_pred):.4f}",
        "R2": f"{r2_score(y_true, y_pred):.4f}",
    }


def train(args):
    result_dir = ensure_dir(args.result) if args.result else ensure_dir("results")
    model_path = Path(args.model) if args.model else default_model_path("linear", "house", "pkl")

    X_train, y_train, X_test, y_test, preprocess_info, _, _ = load_house_splits(random_state=args.random_state)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = _build_metrics(y_test, y_pred)
    prediction_csv = result_dir / f"linear_house_predictions_{model_path.stem.split('_')[-1]}.csv"
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(prediction_csv, index=False, encoding="utf-8-sig")
    save_prediction_scatter(
        y_test,
        y_pred,
        result_dir / f"prediction_linear_house_{model_path.stem.split('_')[-1]}.png",
        "Linear Regression on House Data",
    )

    joblib.dump(
        {
            "model": model,
            "preprocess_info": preprocess_info,
            "metrics": metrics,
        },
        model_path,
    )
    append_metrics("Linear Regression | House", metrics, result_dir)

    print(f"Linear regression model saved to: {model_path}")
    print(f"Prediction table saved to: {prediction_csv}")
    print("Test metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


def test(args):
    result_dir = ensure_dir(args.result) if args.result else ensure_dir("results")
    model_path = Path(args.model) if args.model else find_latest_model("linear_house_", ("pkl",))
    payload = joblib.load(model_path)

    X_train, y_train, X_test, y_test, _, _, _ = load_house_splits(random_state=args.random_state)
    model = payload["model"]
    y_pred = model.predict(X_test)
    metrics = _build_metrics(y_test, y_pred)

    append_metrics("Linear Regression Test | House", metrics, result_dir)
    print(f"Loaded model: {model_path}")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


def predict(args):
    print("Predict mode for house currently reuses the hold-out test split.")
    test(args)


def run(process, args):
    if args.data != "house":
        raise NotImplementedError("Linear regression is only implemented for the house dataset.")

    if process == "train":
        train(args)
    elif process == "test":
        test(args)
    elif process == "predict":
        predict(args)
