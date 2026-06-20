from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier

from data_preprocess import load_image_train_test, load_titanic_train_test
from utils import (
    append_metrics,
    default_model_path,
    ensure_dir,
    find_latest_model,
    plot_accuracy_curve,
    save_confusion_matrix_figure,
    save_prediction_grid,
)

DEFAULT_SAMPLE_LIMITS = {
    "mnist": (4000, 800),
    "cifar10": (3000, 600),
}


def _load_dataset(args):
    if args.data == "titanic":
        X_train, y_train, X_test, y_test, preprocess_info, _, _ = load_titanic_train_test()
        metadata = {
            "dataset_name": "titanic",
            "class_names": ["not survived", "survived"],
            "preprocess_info": preprocess_info,
        }
        return X_train, y_train, X_test, y_test, metadata, None

    max_train, max_test = DEFAULT_SAMPLE_LIMITS[args.data]
    X_train, y_train, X_test, y_test, metadata, _, test_images = load_image_train_test(
        args.data,
        use_hog=False,
        max_train=args.max_train or max_train,
        max_test=args.max_test or max_test,
        seed=args.random_state,
    )
    return X_train, y_train, X_test, y_test, metadata, test_images


def _evaluate_k_values(X_train, y_train, X_test, y_test, k_values):
    scores = []
    for k_value in k_values:
        model = KNeighborsClassifier(n_neighbors=k_value)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores.append(accuracy_score(y_test, y_pred))
    return scores


def train(args):
    result_dir = ensure_dir(args.result) if args.result else ensure_dir("results")
    model_path = Path(args.model) if args.model else default_model_path("knn", args.data, "pkl")

    X_train, y_train, X_test, y_test, metadata, test_images = _load_dataset(args)
    k_values = [args.k] if args.k else [3, 5, 7, 9]
    scores = _evaluate_k_values(X_train, y_train, X_test, y_test, k_values)
    best_index = int(np.argmax(scores))
    best_k = k_values[best_index]

    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    metrics = {
        "best_k": str(best_k),
        "test_accuracy": f"{accuracy:.4f}",
        "train_samples": str(len(y_train)),
        "test_samples": str(len(y_test)),
    }
    append_metrics(f"KNN | {args.data}", metrics, result_dir)
    plot_accuracy_curve(
        k_values,
        scores,
        result_dir / f"loss_curve_knn_{args.data}_{model_path.stem.split('_')[-1]}.png",
        f"KNN Accuracy Curve ({args.data})",
        ylabel="accuracy",
    )
    save_confusion_matrix_figure(
        y_test,
        y_pred,
        metadata["class_names"],
        result_dir / f"prediction_knn_{args.data}_{model_path.stem.split('_')[-1]}.png",
        f"KNN Predictions ({args.data})",
    )

    if test_images is not None:
        save_prediction_grid(
            test_images[:12],
            y_test[:12],
            y_pred[:12],
            metadata["class_names"],
            result_dir / f"sample_knn_{args.data}_{model_path.stem.split('_')[-1]}.png",
            f"KNN Sample Predictions ({args.data})",
        )
    else:
        pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(
            result_dir / f"knn_{args.data}_predictions_{model_path.stem.split('_')[-1]}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    joblib.dump({"model": model, "metadata": metadata}, model_path)
    print(f"KNN model saved to: {model_path}")
    print(classification_report(y_test, y_pred, target_names=metadata["class_names"], zero_division=0))


def test(args):
    result_dir = ensure_dir(args.result) if args.result else ensure_dir("results")
    model_path = Path(args.model) if args.model else find_latest_model(f"knn_{args.data}_", ("pkl",))
    payload = joblib.load(model_path)
    model = payload["model"]
    metadata = payload["metadata"]

    X_train, y_train, X_test, y_test, _, test_images = _load_dataset(args)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    metrics = {
        "model": model_path.name,
        "test_accuracy": f"{accuracy:.4f}",
        "test_samples": str(len(y_test)),
    }
    append_metrics(f"KNN Test | {args.data}", metrics, result_dir)
    save_confusion_matrix_figure(
        y_test,
        y_pred,
        metadata["class_names"],
        result_dir / f"prediction_knn_{args.data}_{model_path.stem.split('_')[-1]}_test.png",
        f"KNN Test ({args.data})",
    )
    if test_images is not None:
        save_prediction_grid(
            test_images[:12],
            y_test[:12],
            y_pred[:12],
            metadata["class_names"],
            result_dir / f"sample_knn_{args.data}_{model_path.stem.split('_')[-1]}_test.png",
            f"KNN Test Samples ({args.data})",
        )
    print(f"Loaded model: {model_path}")
    print(classification_report(y_test, y_pred, target_names=metadata["class_names"], zero_division=0))


def predict(args):
    print("Predict mode currently reuses the test split for KNN.")
    test(args)


def run(process, args):
    if process == "train":
        train(args)
    elif process == "test":
        test(args)
    elif process == "predict":
        predict(args)
