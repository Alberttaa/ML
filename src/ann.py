from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_preprocess import load_house_splits, load_image_train_test, load_titanic_train_test
from utils import (
    append_metrics,
    default_model_path,
    ensure_dir,
    find_latest_model,
    plot_accuracy_curve,
    plot_loss_curve,
    save_confusion_matrix_figure,
    save_prediction_grid,
    save_prediction_scatter,
)


class SimpleImageANN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def _train_house(args):
    result_dir = ensure_dir(args.result) if args.result else ensure_dir("results")
    model_path = Path(args.model) if args.model else default_model_path("ann", "house", "pkl")

    X_train, y_train, X_test, y_test, preprocess_info, _, _ = load_house_splits(random_state=args.random_state)
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=args.random_state,
        learning_rate_init=0.001,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    metrics = {
        "MSE": f"{mse:.4f}",
        "RMSE": f"{mse ** 0.5:.4f}",
        "MAE": f"{mean_absolute_error(y_test, y_pred):.4f}",
        "R2": f"{r2_score(y_test, y_pred):.4f}",
    }
    append_metrics("ANN | House", metrics, result_dir)
    plot_loss_curve(
        model.loss_curve_,
        result_dir / f"loss_curve_ann_house_{model_path.stem.split('_')[-1]}.png",
        "ANN House Loss Curve",
    )
    save_prediction_scatter(
        y_test,
        y_pred,
        result_dir / f"prediction_ann_house_{model_path.stem.split('_')[-1]}.png",
        "ANN House Prediction",
    )
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(
        result_dir / f"ann_house_predictions_{model_path.stem.split('_')[-1]}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    joblib.dump({"model": model, "preprocess_info": preprocess_info}, model_path)
    print(f"ANN house model saved to: {model_path}")


def _test_house(args):
    result_dir = ensure_dir(args.result) if args.result else ensure_dir("results")
    model_path = Path(args.model) if args.model else find_latest_model("ann_house_", ("pkl",))
    payload = joblib.load(model_path)
    model = payload["model"]
    X_train, y_train, X_test, y_test, _, _, _ = load_house_splits(random_state=args.random_state)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    metrics = {
        "model": model_path.name,
        "MSE": f"{mse:.4f}",
        "RMSE": f"{mse ** 0.5:.4f}",
        "MAE": f"{mean_absolute_error(y_test, y_pred):.4f}",
        "R2": f"{r2_score(y_test, y_pred):.4f}",
    }
    append_metrics("ANN Test | House", metrics, result_dir)
    print(metrics)


def _train_titanic(args):
    result_dir = ensure_dir(args.result) if args.result else ensure_dir("results")
    model_path = Path(args.model) if args.model else default_model_path("ann", "titanic", "pkl")

    X_train, y_train, X_test, y_test, preprocess_info, _, _ = load_titanic_train_test()
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=args.random_state,
        learning_rate_init=0.001,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    metrics = {
        "test_accuracy": f"{accuracy:.4f}",
        "train_samples": str(len(y_train)),
        "test_samples": str(len(y_test)),
    }
    append_metrics("ANN | Titanic", metrics, result_dir)
    plot_loss_curve(
        model.loss_curve_,
        result_dir / f"loss_curve_ann_titanic_{model_path.stem.split('_')[-1]}.png",
        "ANN Titanic Loss Curve",
    )
    save_confusion_matrix_figure(
        y_test,
        y_pred,
        ["not survived", "survived"],
        result_dir / f"prediction_ann_titanic_{model_path.stem.split('_')[-1]}.png",
        "ANN Titanic",
    )
    joblib.dump({"model": model, "preprocess_info": preprocess_info}, model_path)
    print(f"ANN titanic model saved to: {model_path}")
    print(classification_report(y_test, y_pred, target_names=["not survived", "survived"], zero_division=0))


def _test_titanic(args):
    result_dir = ensure_dir(args.result) if args.result else ensure_dir("results")
    model_path = Path(args.model) if args.model else find_latest_model("ann_titanic_", ("pkl",))
    payload = joblib.load(model_path)
    model = payload["model"]
    X_train, y_train, X_test, y_test, _, _, _ = load_titanic_train_test()
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    metrics = {
        "model": model_path.name,
        "test_accuracy": f"{accuracy:.4f}",
    }
    append_metrics("ANN Test | Titanic", metrics, result_dir)
    print(classification_report(y_test, y_pred, target_names=["not survived", "survived"], zero_division=0))


def _build_cifar_loaders(args):
    X_train, y_train, X_test, y_test, metadata, _, test_images = load_image_train_test(
        "cifar10",
        use_hog=False,
        max_train=args.max_train or 8000,
        max_test=args.max_test or 2000,
        seed=args.random_state,
    )

    train_tensor = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_tensor = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )
    train_loader = DataLoader(train_tensor, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_tensor, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader, metadata, test_images, X_train.shape[1]


def _train_cifar10(args):
    result_dir = ensure_dir(args.result) if args.result else ensure_dir("results")
    model_path = Path(args.model) if args.model else default_model_path("ann", "cifar10", "pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, metadata, test_images, input_dim = _build_cifar_loaders(args)
    model = SimpleImageANN(input_dim=input_dim, num_classes=len(metadata["class_names"])).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_losses = []
    test_accuracies = []
    best_predictions = None
    best_truth = None

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        total = 0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(labels)
            total += len(labels)

        epoch_loss = running_loss / max(total, 1)
        train_losses.append(epoch_loss)

        model.eval()
        predictions = []
        truths = []
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(device)
                outputs = model(features)
                preds = outputs.argmax(dim=1).cpu().numpy()
                predictions.extend(preds.tolist())
                truths.extend(labels.numpy().tolist())

        accuracy = accuracy_score(truths, predictions)
        test_accuracies.append(accuracy)
        best_predictions = predictions
        best_truth = truths
        print(f"Epoch {epoch + 1}/{args.epochs} | loss={epoch_loss:.4f} | acc={accuracy:.4f}")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "metadata": metadata,
            "input_dim": input_dim,
        },
        model_path,
    )

    metrics = {
        "final_accuracy": f"{test_accuracies[-1]:.4f}",
        "epochs": str(args.epochs),
        "train_samples": str(len(train_loader.dataset)),
        "test_samples": str(len(test_loader.dataset)),
    }
    append_metrics("ANN | CIFAR10", metrics, result_dir)
    plot_loss_curve(
        train_losses,
        result_dir / f"loss_curve_ann_cifar10_{model_path.stem.split('_')[-1]}.png",
        "ANN CIFAR10 Loss Curve",
    )
    plot_accuracy_curve(
        list(range(1, len(test_accuracies) + 1)),
        test_accuracies,
        result_dir / f"accuracy_ann_cifar10_{model_path.stem.split('_')[-1]}.png",
        "ANN CIFAR10 Accuracy Curve",
        xlabel="epoch",
        ylabel="accuracy",
    )
    save_confusion_matrix_figure(
        best_truth,
        best_predictions,
        metadata["class_names"],
        result_dir / f"prediction_ann_cifar10_{model_path.stem.split('_')[-1]}.png",
        "ANN CIFAR10",
    )
    save_prediction_grid(
        test_images[:12],
        np.asarray(best_truth[:12]),
        np.asarray(best_predictions[:12]),
        metadata["class_names"],
        result_dir / f"sample_ann_cifar10_{model_path.stem.split('_')[-1]}.png",
        "ANN CIFAR10 Sample Predictions",
    )
    print(f"ANN CIFAR10 model saved to: {model_path}")


def _test_cifar10(args):
    result_dir = ensure_dir(args.result) if args.result else ensure_dir("results")
    model_path = Path(args.model) if args.model else find_latest_model("ann_cifar10_", ("pth",))
    payload = torch.load(model_path, map_location="cpu")
    metadata = payload["metadata"]
    input_dim = payload["input_dim"]

    _, test_loader, _, test_images, _ = _build_cifar_loaders(args)
    model = SimpleImageANN(input_dim=input_dim, num_classes=len(metadata["class_names"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()

    predictions = []
    truths = []
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            predictions.extend(outputs.argmax(dim=1).numpy().tolist())
            truths.extend(labels.numpy().tolist())

    accuracy = accuracy_score(truths, predictions)
    metrics = {
        "model": model_path.name,
        "test_accuracy": f"{accuracy:.4f}",
    }
    append_metrics("ANN Test | CIFAR10", metrics, result_dir)
    save_confusion_matrix_figure(
        truths,
        predictions,
        metadata["class_names"],
        result_dir / f"prediction_ann_cifar10_{model_path.stem.split('_')[-1]}_test.png",
        "ANN CIFAR10 Test",
    )
    save_prediction_grid(
        test_images[:12],
        np.asarray(truths[:12]),
        np.asarray(predictions[:12]),
        metadata["class_names"],
        result_dir / f"sample_ann_cifar10_{model_path.stem.split('_')[-1]}_test.png",
        "ANN CIFAR10 Test Samples",
    )
    print(classification_report(truths, predictions, target_names=metadata["class_names"], zero_division=0))


def train(args):
    if args.data == "house":
        _train_house(args)
    elif args.data == "titanic":
        _train_titanic(args)
    elif args.data == "cifar10":
        _train_cifar10(args)
    else:
        raise NotImplementedError(f"ANN is not implemented for {args.data}.")


def test(args):
    if args.data == "house":
        _test_house(args)
    elif args.data == "titanic":
        _test_titanic(args)
    elif args.data == "cifar10":
        _test_cifar10(args)
    else:
        raise NotImplementedError(f"ANN is not implemented for {args.data}.")


def predict(args):
    print("Predict mode currently reuses the test split for ANN.")
    test(args)


def run(process, args):
    if process == "train":
        train(args)
    elif process == "test":
        test(args)
    elif process == "predict":
        predict(args)
