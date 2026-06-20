import os
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "data", "house", "house_data.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "house", "house_test.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "linear_house.pkl")
TARGET_COL = "y"


def load_data(file_path):
    return pd.read_csv(file_path)


def preprocess_data(df, is_train=True, preprocess_info=None):
    df = df.copy()

    if TARGET_COL not in df.columns:
        raise ValueError(f"数据中找不到目标列：{TARGET_COL}")

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    if is_train:
        num_imputer = SimpleImputer(strategy="median")
        cat_imputer = SimpleImputer(strategy="most_frequent")

        if numeric_cols:
            X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])

        if categorical_cols:
            X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

        X = pd.get_dummies(X, columns=categorical_cols)

        preprocess_info = {
            "num_imputer": num_imputer,
            "cat_imputer": cat_imputer,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "feature_columns": X.columns.tolist()
        }

        return X, y, preprocess_info

    else:
        if preprocess_info is None:
            raise ValueError("测试阶段必须传入 preprocess_info")

        num_imputer = preprocess_info["num_imputer"]
        cat_imputer = preprocess_info["cat_imputer"]
        numeric_cols = preprocess_info["numeric_cols"]
        categorical_cols = preprocess_info["categorical_cols"]
        feature_columns = preprocess_info["feature_columns"]

        if numeric_cols:
            X[numeric_cols] = num_imputer.transform(X[numeric_cols])

        if categorical_cols:
            X[categorical_cols] = cat_imputer.transform(X[categorical_cols])

        X = pd.get_dummies(X, columns=categorical_cols)

        for col in feature_columns:
            if col not in X.columns:
                X[col] = 0

        X = X[feature_columns]

        return X, y


def train():
    print("正在读取房价训练数据...")
    df_train = load_data(TRAIN_PATH)
    print("训练集列名：", df_train.columns.tolist())
    print("训练集行数：", df_train.shape[0])

    print("正在进行房价数据预处理...")
    X_train, y_train, preprocess_info = preprocess_data(df_train, is_train=True)

    print("开始训练线性回归模型...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    save_obj = {
        "model": model,
        "preprocess_info": preprocess_info
    }

    joblib.dump(save_obj, MODEL_PATH)

    print("线性回归训练完成！")
    print(f"模型已保存到：{MODEL_PATH}")


def test():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("线性回归模型不存在，请先运行：python main.py linear train")

    print("正在加载线性回归模型...")
    save_obj = joblib.load(MODEL_PATH)
    model = save_obj["model"]
    preprocess_info = save_obj["preprocess_info"]

    print("正在读取房价测试数据...")
    df_test = load_data(TEST_PATH)

    print("正在进行房价测试数据预处理...")
    X_test, y_test = preprocess_data(df_test, is_train=False, preprocess_info=preprocess_info)

    print("开始预测...")
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n========== 房价线性回归测试结果 ==========")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.4f}")