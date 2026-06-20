import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, "data", "titanic", "titanic_train_knn.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "titanic", "titanic_test_knn.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "ann_titanic.pkl")
RESULT_DIR = os.path.join(BASE_DIR, "results")

TARGET_COL = "2urvived"


def preprocess_train(df):
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    if numeric_cols:
        X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])
    if categorical_cols:
        X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

    X = pd.get_dummies(X, columns=categorical_cols)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    info = {
        "num_imputer": num_imputer,
        "cat_imputer": cat_imputer,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "feature_columns": X.columns.tolist(),
        "scaler": scaler
    }

    return X_scaled, y, info


def preprocess_test(df, info):
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    numeric_cols = info["numeric_cols"]
    categorical_cols = info["categorical_cols"]

    if numeric_cols:
        X[numeric_cols] = info["num_imputer"].transform(X[numeric_cols])
    if categorical_cols:
        X[categorical_cols] = info["cat_imputer"].transform(X[categorical_cols])

    X = pd.get_dummies(X, columns=categorical_cols)

    for col in info["feature_columns"]:
        if col not in X.columns:
            X[col] = 0

    X = X[info["feature_columns"]]
    X_scaled = info["scaler"].transform(X)

    return X_scaled, y


def train():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("正在读取 Titanic 训练数据...")
    df_train = pd.read_csv(TRAIN_PATH)

    X_train, y_train, info = preprocess_train(df_train)

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
        learning_rate_init=0.001
    )

    print("开始训练 ANN Titanic 模型...")
    model.fit(X_train, y_train)

    joblib.dump({"model": model, "preprocess_info": info}, MODEL_PATH)
    print(f"模型已保存到：{MODEL_PATH}")

    plt.figure(figsize=(8, 5))
    plt.plot(model.loss_curve_)
    plt.xlabel("迭代次数")
    plt.ylabel("Loss")
    plt.title("ANN Titanic训练Loss曲线")
    plt.grid(True)
    loss_path = os.path.join(RESULT_DIR, "ann_titanic_loss.png")
    plt.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Loss曲线已保存到：{loss_path}")


def test():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("模型不存在，请先运行：python main.py ann_titanic train")

    print("正在加载 ANN Titanic 模型...")
    save_obj = joblib.load(MODEL_PATH)
    model = save_obj["model"]
    info = save_obj["preprocess_info"]

    print("正在读取 Titanic 测试数据...")
    df_test = pd.read_csv(TEST_PATH)

    X_test, y_test = preprocess_test(df_test, info)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("\n========== ANN Titanic 测试结果 ==========")
    print(f"准确率：{acc:.4f}")
    print("\n分类报告：")
    print(classification_report(y_test, y_pred))
    print("混淆矩阵：")
    print(confusion_matrix(y_test, y_pred))


def run():
    train()
    test()


if __name__ == "__main__":
    run()