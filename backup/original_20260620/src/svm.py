import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from data_preprocess import load_data, preprocess_data

TRAIN_PATH = r"E:\Ling\homework\aaa\My project\data\titanic\titanic_train_knn.csv"
TEST_PATH = r"E:\Ling\homework\aaa\My project\data\titanic\titanic_test_knn.csv"
MODEL_PATH = r"E:\Ling\homework\aaa\My project\models\svm_titanic.pkl"


def train():
    print("正在读取 Titanic 训练数据...")
    df_train = load_data(TRAIN_PATH)

    print("正在进行数据预处理...")
    X_train, y_train, preprocess_info = preprocess_data(df_train, is_train=True)

    print("开始训练 SVM 模型...")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale"))
    ])

    model.fit(X_train, y_train)

    save_obj = {
        "model": model,
        "preprocess_info": preprocess_info
    }

    joblib.dump(save_obj, MODEL_PATH)

    print("SVM 训练完成！")
    print(f"模型已保存到：{MODEL_PATH}")


def test():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("SVM 模型文件不存在，请先运行：python main.py svm train")

    print("正在加载 SVM 模型...")
    save_obj = joblib.load(MODEL_PATH)
    model = save_obj["model"]
    preprocess_info = save_obj["preprocess_info"]

    print("正在读取 Titanic 测试数据...")
    df_test = load_data(TEST_PATH)

    print("正在进行测试数据预处理...")
    X_test, y_test = preprocess_data(df_test, is_train=False, preprocess_info=preprocess_info)

    print("开始预测...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n========== Titanic SVM 测试结果 ==========")
    print(f"准确率：{acc:.4f}")
    print("\n分类报告：")
    print(report)
    print("混淆矩阵：")
    print(cm)