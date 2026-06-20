import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_path = "C:/Windows/Fonts/msyh.ttc"
my_font = font_manager.FontProperties(fname=font_path)
plt.rcParams["font.family"] = my_font.get_name()
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "ann", "分类.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

LR = 0.005
EPOCHS = 300

df = pd.read_csv(DATA_PATH)
df = df.dropna()
def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def binary_cross_entropy(y_pred, y_true):
    eps = 1e-8
    return -(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))


def init_params():
    np.random.seed(42)
    W1 = np.random.randn(4, 5) * 0.1
    b1 = np.zeros((4, 1))

    W2 = np.random.randn(3, 4) * 0.1
    b2 = np.zeros((3, 1))

    W3 = np.random.randn(1, 3) * 0.1
    b3 = np.zeros((1, 1))

    return W1, b1, W2, b2, W3, b3


def forward(x, W1, b1, W2, b2, W3, b3):
    z1 = W1 @ x + b1
    a1 = relu(z1)

    z2 = W2 @ a1 + b2
    a2 = relu(z2)

    z3 = W3 @ a2 + b3
    y_pred = sigmoid(z3)

    cache = {
        "x": x,
        "z1": z1,
        "a1": a1,
        "z2": z2,
        "a2": a2,
        "z3": z3,
        "y_pred": y_pred
    }

    return y_pred, cache


def backward(y_true, cache, W2, W3):
    x = cache["x"]
    z1 = cache["z1"]
    a1 = cache["a1"]
    z2 = cache["z2"]
    a2 = cache["a2"]
    y_pred = cache["y_pred"]

    # sigmoid + 交叉熵的简化梯度
    dz3 = y_pred - y_true
    dW3 = dz3 @ a2.T
    db3 = dz3

    da2 = W3.T @ dz3
    dz2 = da2 * relu_derivative(z2)
    dW2 = dz2 @ a1.T
    db2 = dz2

    da1 = W2.T @ dz2
    dz1 = da1 * relu_derivative(z1)
    dW1 = dz1 @ x.T
    db1 = dz1

    return dW1, db1, dW2, db2, dW3, db3


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("正在读取分类数据：", DATA_PATH)

    df = pd.read_csv(DATA_PATH)

    # 删除空值行
    df = df.dropna()

    x_cols = ["(x_1)", "(x_2)", "(x_3)", "(x_4)", "(x_5)"]
    y_col = "(y)"

    X = df[x_cols].values.astype(float)
    y = df[y_col].values.astype(float)

    W1, b1, W2, b2, W3, b3 = init_params()

    loss_history = []
    acc_history = []

    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0

        for i in range(len(X)):
            x_i = X[i].reshape(5, 1)
            y_i = y[i]

            y_pred, cache = forward(x_i, W1, b1, W2, b2, W3, b3)

            loss = binary_cross_entropy(y_pred[0, 0], y_i)
            total_loss += loss

            pred_label = 1 if y_pred[0, 0] >= 0.5 else 0
            if pred_label == int(y_i):
                correct += 1

            dW1, db1, dW2, db2, dW3, db3 = backward(y_i, cache, W2, W3)

            W1 -= LR * dW1
            b1 -= LR * db1
            W2 -= LR * dW2
            b2 -= LR * db2
            W3 -= LR * dW3
            b3 -= LR * db3

        avg_loss = total_loss / len(X)
        acc = correct / len(X)

        loss_history.append(avg_loss)
        acc_history.append(acc)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS}, Loss={avg_loss:.4f}, Accuracy={acc:.4f}")

    print("\n========== ANN 二分类结果 ==========")
    print(f"最终 Loss：{loss_history[-1]:.4f}")
    print(f"最终 Accuracy：{acc_history[-1]:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(loss_history)
    plt.xlabel("训练轮数 Epoch")
    plt.ylabel("Loss")
    plt.title("ANN二分类Loss曲线")
    plt.grid(True)
    loss_path = os.path.join(RESULTS_DIR, "manual_ann_classification_loss.png")
    plt.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(acc_history)
    plt.xlabel("训练轮数 Epoch")
    plt.ylabel("Accuracy")
    plt.title("ANN二分类Accuracy曲线")
    plt.grid(True)
    acc_path = os.path.join(RESULTS_DIR, "manual_ann_classification_accuracy.png")
    plt.savefig(acc_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("Loss曲线保存到：", loss_path)
    print("Accuracy曲线保存到：", acc_path)


if __name__ == "__main__":
    run()