from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"


def ensure_dir(path):
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def today_tag():
    return datetime.now().strftime("%Y%m%d")


def default_model_path(algo, dataset, extension):
    ensure_dir(MODELS_DIR)
    return MODELS_DIR / f"{algo}_{dataset}_{today_tag()}.{extension}"


def find_latest_model(prefix, extensions):
    ensure_dir(MODELS_DIR)
    candidates = []
    for extension in extensions:
        candidates.extend(MODELS_DIR.glob(f"{prefix}*.{extension}"))
    if not candidates:
        raise FileNotFoundError(f"No model found with prefix '{prefix}' under {MODELS_DIR}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def append_metrics(title, metrics, result_dir=None):
    result_dir = ensure_dir(result_dir or RESULTS_DIR)
    metrics_path = result_dir / "accuracy.txt"
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(f"===== {title} =====\n")
        for key, value in metrics.items():
            handle.write(f"{key}: {value}\n")
        handle.write("\n")


def plot_loss_curve(values, output_path, title):
    if not values:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(values, linewidth=2)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_accuracy_curve(x_values, y_values, output_path, title, xlabel="k", ylabel="accuracy"):
    if not x_values:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, marker="o", linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_prediction_scatter(y_true, y_pred, output_path, title):
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    line_min = float(min(np.min(y_true), np.min(y_pred)))
    line_max = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([line_min, line_max], [line_min, line_max], linestyle="--", color="black")
    plt.xlabel("true")
    plt.ylabel("pred")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_confusion_matrix_figure(y_true, y_pred, class_names, output_path, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("predicted")
    plt.ylabel("true")

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            plt.text(col, row, cm[row, col], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_prediction_grid(images, y_true, y_pred, class_names, output_path, title, max_items=12):
    if images is None or len(images) == 0:
        return

    item_count = min(max_items, len(images))
    cols = 4
    rows = int(np.ceil(item_count / cols))
    plt.figure(figsize=(cols * 3, rows * 3))

    for index in range(item_count):
        plt.subplot(rows, cols, index + 1)
        image = images[index]
        if image.ndim == 2:
            plt.imshow(image, cmap="gray")
        else:
            plt.imshow(image.astype(np.uint8))
        plt.title(f"T:{class_names[int(y_true[index])]}\nP:{class_names[int(y_pred[index])]}")
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
