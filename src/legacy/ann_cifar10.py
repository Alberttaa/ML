import os
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import cifar10
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(BASE_DIR, "results")


def run():
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("正在加载 CIFAR10 数据...")
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # 为了避免运行太慢，先取部分数据
    X_train = X_train[:5000]
    y_train = y_train[:5000]
    X_test = X_test[:1000]
    y_test = y_test[:1000]

    # 图片拉平成向量，并归一化到 0~1
    X_train = X_train.reshape(len(X_train), -1) / 255.0
    X_test = X_test.reshape(len(X_test), -1) / 255.0

    model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        max_iter=50,
        random_state=42,
        learning_rate_init=0.001
    )

    print("开始训练 ANN CIFAR10 分类模型...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n========== ANN CIFAR10 分类结果 ==========")
    print(f"准确率：{acc:.4f}")
    print("\n分类报告：")
    print(classification_report(y_test, y_pred))

    plt.figure(figsize=(8, 5))
    plt.plot(model.loss_curve_)
    plt.xlabel("迭代次数")
    plt.ylabel("Loss")
    plt.title("ANN CIFAR10训练Loss曲线")
    plt.grid(True)
    loss_path = os.path.join(RESULT_DIR, "ann_cifar10_loss.png")
    plt.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Loss曲线已保存到：{loss_path}")


if __name__ == "__main__":
    run()