import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "house", "house_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "ann_house.pkl")
RESULT_DIR = os.path.join(BASE_DIR, "results")

TARGET_COL = "y"


def run():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("正在读取房价数据...")
    df = pd.read_csv(DATA_PATH)
    print("数据列名：", df.columns.tolist())

    if TARGET_COL not in df.columns:
        raise ValueError(f"找不到目标列：{TARGET_COL}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=42,
        learning_rate_init=0.001
    )

    print("开始训练 ANN 房价预测模型...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n========== ANN 房价预测结果 ==========")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.4f}")

    joblib.dump({"model": model, "scaler": scaler}, MODEL_PATH)
    print(f"模型已保存到：{MODEL_PATH}")

    plt.figure(figsize=(8, 5))
    plt.plot(model.loss_curve_)
    plt.xlabel("迭代次数")
    plt.ylabel("Loss")
    plt.title("ANN房价预测Loss曲线")
    plt.grid(True)
    loss_path = os.path.join(RESULT_DIR, "ann_house_loss.png")
    plt.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Loss曲线已保存到：{loss_path}")


if __name__ == "__main__":
    run()