import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# =========================
# 1. 读取图片并提取 HOG 特征
# =========================
def load_mnist_hog_data(base_path, img_size=(28, 28)):
    X = []
    y = []

    for label_name in ['0', '1']:
        folder_path = os.path.join(base_path, label_name)
        label = int(label_name)

        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            # 读取灰度图
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # 统一尺寸
            img = cv2.resize(img, img_size)

            # 提取 HOG 特征
            features = hog(
                img,
                orientations=9,
                pixels_per_cell=(4, 4),
                cells_per_block=(2, 2),
                block_norm='L2-Hys'
            )

            X.append(features)
            y.append(label)

    X = np.array(X)
    y = np.array(y)
    return X, y


# =========================
# 2. sigmoid 函数
# =========================
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


# =========================
# 3. 计算 loss（带 L2 正则）
# =========================
def compute_loss(X, y, w, b, lambda_reg=0.01):
    m = X.shape[0]
    z = np.dot(X, w) + b
    y_hat = sigmoid(z)

    eps = 1e-8
    loss = -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))
    l2_loss = (lambda_reg / (2 * m)) * np.sum(w ** 2)
    return loss + l2_loss


# =========================
# 4. 手写逻辑回归训练
# =========================
def train_logistic_regression(X, y, lr=0.01, epochs=100, batch_size=64, lambda_reg=0.01):
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0
    loss_history = []

    for epoch in range(epochs):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            batch_m = X_batch.shape[0]

            z = np.dot(X_batch, w) + b
            y_hat = sigmoid(z)

            dw = (1 / batch_m) * np.dot(X_batch.T, (y_hat - y_batch)) + (lambda_reg / m) * w
            db = (1 / batch_m) * np.sum(y_hat - y_batch)

            w -= lr * dw
            b -= lr * db

        loss = compute_loss(X, y, w, b, lambda_reg)
        loss_history.append(loss)

        print(f"Epoch {epoch+1}/{epochs}, Loss = {loss:.6f}")

    return w, b, loss_history


# =========================
# 5. 预测函数
# =========================
def predict(X, w, b):
    probs = sigmoid(np.dot(X, w) + b)
    return (probs >= 0.5).astype(int)


# =========================
# 6. 主程序
# =========================
if __name__ == "__main__":
    base_path = "."   # 你的 mnist 文件夹路径

    # 读取数据
    X, y = load_mnist_hog_data(base_path)
    print("数据形状:", X.shape)
    print("标签形状:", y.shape)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 特征标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 训练逻辑回归
    w, b, loss_history = train_logistic_regression(
        X_train, y_train,
        lr=0.01,
        epochs=100,
        batch_size=64,
        lambda_reg=0.01
    )

    # 画 loss 曲线
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.grid(True)
    plt.show()

    # 测试集预测
    y_pred = predict(X_test, w, b)

    # 输出测试准确率
    acc = accuracy_score(y_test, y_pred)
    print("测试准确率:", acc)