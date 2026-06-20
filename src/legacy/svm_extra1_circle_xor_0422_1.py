import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# ======================================================
# 附加题1简略思路：
# 1. 分别读取 circle_data.csv 和 xor_dataset.csv；
# 2. 每个数据集取前两列作为特征 X，最后一列作为标签 y；
# 3. 按题目要求：训练集和测试集相同；
# 4. 分别训练逻辑回归和 SVM；
# 5. 输出两种算法在同一数据集上的准确率；
# 6. 画出分类结果和决策边界，方便对比。
# ======================================================


# =========================
# 1. 路径配置
# =========================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

CIRCLE_PATH = r"E:\Ling\homework\aaa\My project\data\svm\circles_dataset.csv"
XOR_PATH = r"E:\Ling\homework\aaa\My project\data\svm\xor_dataset.csv"


# =========================
# 2. 创建结果文件夹
# =========================

def make_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================
# 3. 读取 CSV 数据
# =========================

def read_csv_safely(path):
    """
    读取 CSV，兼容中文路径和常见编码。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件：{path}")

    for enc in ["utf-8", "gbk", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue

    return pd.read_csv(path)


def load_dataset(path):
    """
    默认处理方式：
    - 前两列作为特征 X
    - 最后一列作为标签 y

    如果你的 csv 有列名，比如 x1,x2,y，也可以正常读取。
    """
    df = read_csv_safely(path)

    print("\n读取文件：", path)
    print("数据列名：", list(df.columns))
    print("数据形状：", df.shape)

    # 去掉空值
    df = df.dropna()

    # 如果标签列名字比较常见，就优先识别
    possible_label_cols = ["label", "Label", "y", "Y", "target", "Target", "class", "Class"]

    label_col = None
    for col in possible_label_cols:
        if col in df.columns:
            label_col = col
            break

    if label_col is not None:
        y = df[label_col].values
        X = df.drop(columns=[label_col]).values
    else:
        # 否则默认最后一列是标签
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

    # 只取前两个特征，方便画二维决策边界
    X = X[:, :2]

    # 标签转成整数
    y = y.astype(int)

    print("X 维度：", X.shape)
    print("y 维度：", y.shape)
    print("标签类别：", np.unique(y))

    return X, y


# =========================
# 4. 构建模型
# =========================

def build_models():
    """
    逻辑回归：线性分类器
    SVM：使用 RBF 核，适合非线性分类
    """

    logistic_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    svm_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=10.0, gamma="scale"))
    ])

    models = {
        "Logistic Regression": logistic_model,
        "SVM RBF": svm_model
    }

    return models


# =========================
# 5. 训练和测试
# =========================

def train_and_evaluate(X, y, dataset_name):
    """
    按题目要求：训练集和测试集相同。
    所以这里直接 model.fit(X, y)，再 model.predict(X)。
    """

    models = build_models()
    result = {}

    print("\n" + "=" * 60)
    print(f"开始实验：{dataset_name}")
    print("=" * 60)

    for model_name, model in models.items():
        model.fit(X, y)

        y_pred = model.predict(X)

        acc = accuracy_score(y, y_pred)
        result[model_name] = {
            "model": model,
            "acc": acc,
            "y_pred": y_pred
        }

        print(f"{dataset_name} | {model_name} 准确率：{acc:.4f}")

    return result


# =========================
# 6. 绘制决策边界
# =========================

def plot_decision_boundary(X, y, model, dataset_name, model_name, acc):
    """
    画二维分类边界：
    背景颜色表示模型预测类别；
    散点表示真实样本类别。
    """

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(7, 6))

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", s=35)

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(f"{dataset_name} - {model_name} | Acc={acc:.4f}")
    plt.tight_layout()

    save_name = f"{dataset_name}_{model_name}_boundary.png"
    save_name = save_name.replace(" ", "_").replace("/", "_")
    save_path = os.path.join(RESULTS_DIR, save_name)

    plt.savefig(save_path, dpi=300)
    print("决策边界图已保存：", save_path)

    plt.show()


# =========================
# 7. 绘制准确率对比图
# =========================

def plot_accuracy_bar(all_results):
    """
    对比 circle 和 xor 上逻辑回归、SVM 的准确率。
    """

    dataset_names = []
    lr_accs = []
    svm_accs = []

    for dataset_name, results in all_results.items():
        dataset_names.append(dataset_name)
        lr_accs.append(results["Logistic Regression"]["acc"])
        svm_accs.append(results["SVM RBF"]["acc"])

    x = np.arange(len(dataset_names))
    width = 0.35

    plt.figure(figsize=(8, 5))

    plt.bar(x - width / 2, lr_accs, width, label="Logistic Regression")
    plt.bar(x + width / 2, svm_accs, width, label="SVM RBF")

    plt.xticks(x, dataset_names)
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison on Circle and XOR Dataset")
    plt.legend()

    for i, acc in enumerate(lr_accs):
        plt.text(i - width / 2, acc + 0.02, f"{acc:.2f}", ha="center")

    for i, acc in enumerate(svm_accs):
        plt.text(i + width / 2, acc + 0.02, f"{acc:.2f}", ha="center")

    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, "svm_logistic_accuracy_comparison.png")
    plt.savefig(save_path, dpi=300)
    print("准确率对比图已保存：", save_path)

    plt.show()


# =========================
# 8. 主函数
# =========================

def main():
    make_dirs()

    datasets = {
        "circle_data": CIRCLE_PATH,
        "xor_dataset": XOR_PATH
    }

    all_results = {}

    for dataset_name, path in datasets.items():
        X, y = load_dataset(path)

        results = train_and_evaluate(X, y, dataset_name)
        all_results[dataset_name] = results

        # 分别画逻辑回归和 SVM 的决策边界
        for model_name, info in results.items():
            plot_decision_boundary(
                X=X,
                y=y,
                model=info["model"],
                dataset_name=dataset_name,
                model_name=model_name,
                acc=info["acc"]
            )

    plot_accuracy_bar(all_results)

    print("\n========== 最终结果汇总 ==========")
    for dataset_name, results in all_results.items():
        print(f"\n数据集：{dataset_name}")
        for model_name, info in results.items():
            print(f"{model_name}: acc = {info['acc']:.4f}")


if __name__ == "__main__":
    main()