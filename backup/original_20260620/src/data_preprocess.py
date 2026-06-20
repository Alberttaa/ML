import pandas as pd
from sklearn.impute import SimpleImputer

# 你的数据里标签列名不是 Survived，而是 2urvived
TARGET_COL = "2urvived"


def load_data(file_path):
    """
    读取CSV文件
    """
    df = pd.read_csv(file_path)
    return df


def preprocess_data(df, is_train=True, preprocess_info=None):
    """
    数据预处理
    功能：
    1. 分离特征和标签
    2. 缺失值填补
    3. 类别特征独热编码
    4. 保证测试集特征列与训练集一致

    参数：
    - df: 原始DataFrame
    - is_train: 是否为训练阶段
    - preprocess_info: 测试阶段需要传入训练阶段保存的预处理信息

    返回：
    训练阶段：
        X, y, preprocess_info
    测试阶段：
        X, y
    """
    df = df.copy()

    if TARGET_COL not in df.columns:
        raise ValueError(f"数据中找不到标签列：{TARGET_COL}")

    # 分离标签和特征
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    # 自动识别数值列和类别列
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    if is_train:
        # 数值特征：用中位数填补
        num_imputer = SimpleImputer(strategy="median")
        # 类别特征：用众数填补
        cat_imputer = SimpleImputer(strategy="most_frequent")

        if numeric_cols:
            X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])

        if categorical_cols:
            X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

        # 独热编码
        X = pd.get_dummies(X, columns=categorical_cols)

        # 保存训练阶段的预处理信息，供测试阶段复用
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

        # 用训练阶段的填补器处理测试集
        if numeric_cols:
            X[numeric_cols] = num_imputer.transform(X[numeric_cols])

        if categorical_cols:
            X[categorical_cols] = cat_imputer.transform(X[categorical_cols])

        # 测试集做独热编码
        X = pd.get_dummies(X, columns=categorical_cols)

        # 保证测试集列和训练集一致
        for col in feature_columns:
            if col not in X.columns:
                X[col] = 0

        # 多余列去掉，并按训练集顺序排列
        X = X[feature_columns]

        return X, y