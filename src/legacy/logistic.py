import os
import io
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# =========================
# 路径设置
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "data", "titanic", "titanic_train.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "titanic", "titanic_test.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

TARGET_COL = "2urvived"


# =========================
# 手写逻辑回归
# =========================
class MyLogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=500, batch_size=64, lambda_l2=0.05, random_state=42):
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.lambda_l2 = lambda_l2
        self.random_state = random_state
        self.loss_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -10, 10)))

    def compute_loss(self, X, y):
        m = len(y)
        y_pred = self.sigmoid(np.dot(X, self.w) + self.b)
        cross_entropy = -np.mean(
            y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8)
        )
        l2_reg = (self.lambda_l2 / (2 * m)) * np.sum(self.w ** 2)
        return cross_entropy + l2_reg

    def fit(self, X, y):
        np.random.seed(self.random_state)
        m, n = X.shape
        self.w = np.random.normal(0, 0.01, n)
        self.b = 0

        for epoch in range(self.epochs):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0

            for i in range(0, m, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                z = np.dot(X_batch, self.w) + self.b
                y_pred = self.sigmoid(z)

                dw = (1 / len(X_batch)) * np.dot(X_batch.T, (y_pred - y_batch)) + \
                     (self.lambda_l2 / len(X_batch)) * self.w
                db = np.mean(y_pred - y_batch)

                self.w -= self.lr * dw
                self.b -= self.lr * db

                batch_loss = self.compute_loss(X_batch, y_batch)
                epoch_loss += batch_loss * (len(X_batch) / m)

            self.loss_history.append(epoch_loss)

            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.4f}")

    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        y_pred = self.sigmoid(z)
        return (y_pred >= 0.5).astype(int)


# =========================
# 数据读取与预处理
# =========================
def load_and_preprocess_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # 删除全0列
    zero_cols = [col for col in train_df.columns if (train_df[col] == 0).all()]
    train_df = train_df.drop(columns=zero_cols)
    test_df = test_df.drop(columns=zero_cols, errors="ignore")

    # 缺失值与异常值处理
    train_df["Age"] = train_df["Age"].fillna(train_df["Age"].median())
    test_df["Age"] = test_df["Age"].fillna(test_df["Age"].median())

    fare_95_train = train_df["Fare"].quantile(0.95)
    train_df["Fare"] = train_df["Fare"].apply(lambda x: fare_95_train if x > fare_95_train else x)
    train_df["Fare"] = train_df["Fare"].fillna(train_df["Fare"].median())

    fare_95_test = test_df["Fare"].quantile(0.95)
    test_df["Fare"] = test_df["Fare"].apply(lambda x: fare_95_test if x > fare_95_test else x)
    test_df["Fare"] = test_df["Fare"].fillna(test_df["Fare"].median())

    train_df["Embarked"] = train_df["Embarked"].fillna(train_df["Embarked"].mode()[0])
    test_df["Embarked"] = test_df["Embarked"].fillna(test_df["Embarked"].mode()[0])

    # One-Hot 编码
    encoder = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")

    train_encoded = encoder.fit_transform(train_df[["Pclass", "Embarked", "Sex"]])
    train_encoded_df = pd.DataFrame(
        train_encoded,
        columns=encoder.get_feature_names_out(["Pclass", "Embarked", "Sex"])
    )

    test_encoded = encoder.transform(test_df[["Pclass", "Embarked", "Sex"]])
    test_encoded_df = pd.DataFrame(
        test_encoded,
        columns=encoder.get_feature_names_out(["Pclass", "Embarked", "Sex"])
    )

    train_df = pd.concat(
        [train_df.drop(["Pclass", "Embarked", "Sex"], axis=1).reset_index(drop=True),
         train_encoded_df.reset_index(drop=True)],
        axis=1
    )

    test_df = pd.concat(
        [test_df.drop(["Pclass", "Embarked", "Sex"], axis=1).reset_index(drop=True),
         test_encoded_df.reset_index(drop=True)],
        axis=1
    )

    X_train = train_df.drop(["Passengerid", TARGET_COL], axis=1)
    y_train = train_df[TARGET_COL].astype(int)

    X_test = test_df.drop(["Passengerid", TARGET_COL], axis=1)
    y_test = test_df[TARGET_COL].astype(int)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    preprocess_info = {
        "encoder": encoder,
        "scaler": scaler
    }

    return X_train, y_train, X_test, y_test, preprocess_info


# =========================
# 训练
# =========================
def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("正在读取并预处理 Titanic 数据...")
    X_train, y_train, X_test, y_test, preprocess_info = load_and_preprocess_data()

    print("开始训练 sklearn 逻辑回归...")
    sklearn_model = LogisticRegression(
        penalty="l2",
        C=0.8,
        solver="liblinear",
        max_iter=1000,
        random_state=42
    )
    sklearn_model.fit(X_train, y_train)

    print("开始训练手写 Mini-Batch + L2 逻辑回归...")
    my_model = MyLogisticRegression(
        learning_rate=0.01,
        epochs=500,
        batch_size=64,
        lambda_l2=0.05,
        random_state=42
    )
    my_model.fit(X_train, y_train)

    # 保存 sklearn 模型与预处理器
    model_path = os.path.join(MODELS_DIR, "logistic_titanic.pkl")
    joblib.dump(
        {
            "model": sklearn_model,
            "preprocess_info": preprocess_info
        },
        model_path
    )

    # 保存手写模型参数
    my_model_path = os.path.join(MODELS_DIR, "logistic_titanic_manual.npz")
    np.savez(
        my_model_path,
        w=my_model.w,
        b=my_model.b,
        loss_history=np.array(my_model.loss_history)
    )

    # 保存 loss 图
    plt.figure(figsize=(10, 5))
    plt.plot(my_model.loss_history, linewidth=2)
    plt.xlabel("迭代次数 (Epochs)")
    plt.ylabel("损失值 (Loss)")
    plt.title("Mini-Batch 逻辑回归损失曲线 (带L2正则)")
    plt.grid(True, alpha=0.3)
    loss_path = os.path.join(RESULTS_DIR, "logistic_titanic_loss.png")
    plt.tight_layout()
    plt.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("逻辑回归训练完成！")
    print(f"sklearn模型已保存到：{model_path}")
    print(f"手写模型参数已保存到：{my_model_path}")
    print(f"Loss曲线已保存到：{loss_path}")


# =========================
# 测试
# =========================
def test():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, "logistic_titanic.pkl")
    my_model_path = os.path.join(MODELS_DIR, "logistic_titanic_manual.npz")

    if not os.path.exists(model_path):
        raise FileNotFoundError("未找到 sklearn 逻辑回归模型，请先运行 train()")
    if not os.path.exists(my_model_path):
        raise FileNotFoundError("未找到手写逻辑回归参数，请先运行 train()")

    print("正在读取并预处理 Titanic 数据...")
    X_train, y_train, X_test, y_test, preprocess_info = load_and_preprocess_data()

    # 加载 sklearn 模型
    save_obj = joblib.load(model_path)
    sklearn_model = save_obj["model"]

    # 加载手写模型参数
    data = np.load(my_model_path)
    my_model = MyLogisticRegression()
    my_model.w = data["w"]
    my_model.b = float(data["b"])
    my_model.loss_history = data["loss_history"].tolist()

    # sklearn 模型评估
    y_pred_train = sklearn_model.predict(X_train)
    y_pred_test = sklearn_model.predict(X_test)

    acc_train_sklearn = accuracy_score(y_train, y_pred_train)
    acc_test_sklearn = accuracy_score(y_test, y_pred_test)

    print("===== Sklearn 内置逻辑回归 =====")
    print(f"训练集准确率: {acc_train_sklearn:.4f}")
    print(f"测试集准确率: {acc_test_sklearn:.4f}")

    # 手写模型评估
    y_pred_train_my = my_model.predict(X_train)
    y_pred_test_my = my_model.predict(X_test)

    acc_train_my = accuracy_score(y_train, y_pred_train_my)
    acc_test_my = accuracy_score(y_test, y_pred_test_my)

    print("\n===== 手写 Mini-Batch + L2 逻辑回归 =====")
    print(f"训练集准确率: {acc_train_my:.4f}")
    print(f"测试集准确率: {acc_test_my:.4f}")

    print("\n===== 模型准确率对比 =====")
    print(f"Sklearn模型 - 训练集: {acc_train_sklearn:.4f}, 测试集: {acc_test_sklearn:.4f}")
    print(f"手写模型   - 训练集: {acc_train_my:.4f}, 测试集: {acc_test_my:.4f}")

    accuracy_txt = os.path.join(RESULTS_DIR, "accuracy.txt")
    with open(accuracy_txt, "a", encoding="utf-8") as f:
        f.write("===== Logistic Titanic 二分类 =====\n")
        f.write(f"Sklearn 训练集准确率: {acc_train_sklearn:.4f}\n")
        f.write(f"Sklearn 测试集准确率: {acc_test_sklearn:.4f}\n")
        f.write(f"手写模型训练集准确率: {acc_train_my:.4f}\n")
        f.write(f"手写模型测试集准确率: {acc_test_my:.4f}\n\n")

    print(f"\n准确率摘要已追加保存到：{accuracy_txt}")


def run():
    train()
    test()


if __name__ == "__main__":
    run()