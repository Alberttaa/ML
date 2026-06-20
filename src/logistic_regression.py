from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from data_preprocess import load_image_train_test, load_titanic_train_test
from utils import (
    append_metrics,
    default_model_path,
    ensure_dir,
    find_latest_model,
    save_confusion_matrix_figure,
    save_prediction_grid,
)

DEFAULT_BINARY_CLASSES = {
    "mnist": ("0", "1"),
    "cifar10": ("airplane", "automobile"),
}


def _selected_classes(args):
    if args.data == "titanic":
        return None
    default_a, default_b = DEFAULT_BINARY_CLASSES[args.data]
    return [args.class_a or default_a, args.class_b or default_b]


def _load_dataset(args):
    if args.data == "titanic":
        X_train, y_train, X_test, y_test, preprocess_info, _, _ = load_titanic_train_test()
        metadata = {
            "dataset_name": "titanic",
            "class_names": ["not survived", "survived"],
            "preprocess_info": preprocess_info,
            "selected_classes": None,
            "use_hog": False,
        }
        return X_train, y_train, X_test, y_test, metadata, None

    selected = _selected_classes(args)
    max_train = args.max_train or 4000
    max_test = args.max_test or 800
    X_train, y_train, X_test, y_test, metadata, _, test_images = load_image_train_test(
        args.data,
        use_hog=True,
        selected_classes=selected,
        max_train=max_train,
        max_test=max_test,
        seed=args.random_state,
    )
    return X_train, y_train, X_test, y_test, metadata, test_images


def train(args):
    result_dir = ensure_dir(args.result) if args.result else ensure_dir("results")
    model_path = Path(args.model) if args.model else default_model_path("logistic", args.data, "pkl")
    X_train, y_train, X_test, y_test, metadata, test_images = _load_dataset(args)

    model = LogisticRegression(max_iter=1200, solver="liblinear", random_state=args.random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    metrics = {
        "test_accuracy": f"{accuracy:.4f}",
        "train_samples": str(len(y_train)),
        "test_samples": str(len(y_test)),
    }
    if metadata["selected_classes"]:
        metrics["selected_classes"] = ", ".join(metadata["selected_classes"])
    append_metrics(f"Logistic Regression | {args.data}", metrics, result_dir)
    save_confusion_matrix_figure(
        y_test,
        y_pred,
        metadata["class_names"],
        result_dir / f"prediction_logistic_{args.data}_{model_path.stem.split('_')[-1]}.png",
        f"Logistic Regression ({args.data})",
    )
    if test_images is not None:
        save_prediction_grid(
            test_images[:12],
            y_test[:12],
            y_pred[:12],
            metadata["class_names"],
            result_dir / f"sample_logistic_{args.data}_{model_path.stem.split('_')[-1]}.png",
            f"Logistic Sample Predictions ({args.data})",
        )

    joblib.dump({"model": model, "metadata": metadata}, model_path)
    print(f"Logistic regression model saved to: {model_path}")
    print(classification_report(y_test, y_pred, target_names=metadata["class_names"], zero_division=0))


def test(args):
    result_dir = ensure_dir(args.result) if args.result else ensure_dir("results")
    model_path = Path(args.model) if args.model else find_latest_model(f"logistic_{args.data}_", ("pkl",))
    payload = joblib.load(model_path)
    model = payload["model"]
    metadata = payload["metadata"]

    if args.data == "titanic":
        X_train, y_train, X_test, y_test, _, _, _ = load_titanic_train_test()
        test_images = None
    else:
        X_train, y_train, X_test, y_test, _, _, test_images = load_image_train_test(
            args.data,
            use_hog=True,
            selected_classes=metadata["selected_classes"],
            max_train=args.max_train or 4000,
            max_test=args.max_test or 800,
            seed=args.random_state,
        )

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    metrics = {
        "model": model_path.name,
        "test_accuracy": f"{accuracy:.4f}",
    }
    append_metrics(f"Logistic Regression Test | {args.data}", metrics, result_dir)
    save_confusion_matrix_figure(
        y_test,
        y_pred,
        metadata["class_names"],
        result_dir / f"prediction_logistic_{args.data}_{model_path.stem.split('_')[-1]}_test.png",
        f"Logistic Regression Test ({args.data})",
    )
    if test_images is not None:
        save_prediction_grid(
            test_images[:12],
            y_test[:12],
            y_pred[:12],
            metadata["class_names"],
            result_dir / f"sample_logistic_{args.data}_{model_path.stem.split('_')[-1]}_test.png",
            f"Logistic Test Samples ({args.data})",
        )
    print(f"Loaded model: {model_path}")
    print(classification_report(y_test, y_pred, target_names=metadata["class_names"], zero_division=0))


def predict(args):
    print("Predict mode currently reuses the test split for logistic regression.")
    test(args)


def run(process, args):
    if process == "train":
        train(args)
    elif process == "test":
        test(args)
    elif process == "predict":
        predict(args)
