import numpy as np
import matplotlib.pyplot as plt
import time

from keras.datasets import cifar10
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# 0. 参数设置
# =========================
K = 5

# 为了避免运行太慢，先用子集实验
# 如果电脑性能还可以，可以把这些数调大
TRAIN_SIZE = 5000
TEST_SIZE = 1000

# PCA降维维数
PCA_DIM = 100

# HOG参数
PIXELS_PER_CELL = (4, 4)
CELLS_PER_BLOCK = (2, 2)
ORIENTATIONS = 9

# 是否显示部分样本图
SHOW_SAMPLES = True

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# =========================
# 1. 读取 CIFAR-10 数据
# =========================
print("开始加载 CIFAR-10 数据集...")
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
y_train = y_train.flatten()
y_test = y_test.flatten()

print("原始训练集形状：", X_train.shape)
print("原始测试集形状：", X_test.shape)

# 截取子集，提升运行速度
X_train = X_train[:TRAIN_SIZE]
y_train = y_train[:TRAIN_SIZE]
X_test = X_test[:TEST_SIZE]
y_test = y_test[:TEST_SIZE]

print("实际使用训练集形状：", X_train.shape)
print("实际使用测试集形状：", X_test.shape)

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# =========================
# 2. 显示部分样本
# =========================
if SHOW_SAMPLES:
    plt.figure(figsize=(10, 4))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_train[i])
        plt.title(class_names[y_train[i]])
        plt.axis("off")
    plt.suptitle("CIFAR-10 部分训练样本")
    plt.tight_layout()
    plt.show()


# =========================
# 3. 图像转灰度
# =========================
print("\n开始将彩色图像转换为灰度图...")
start_time = time.time()

X_train_gray = np.array([rgb2gray(img) for img in X_train], dtype=np.float32)
X_test_gray = np.array([rgb2gray(img) for img in X_test], dtype=np.float32)

print("灰度训练集形状：", X_train_gray.shape)
print("灰度测试集形状：", X_test_gray.shape)
print("灰度转换耗时：{:.2f} 秒".format(time.time() - start_time))


# =========================
# 4. 提取 HOG 特征
# =========================
def extract_hog_features(images):
    features = []
    for img in images:
        feat = hog(
            img,
            orientations=ORIENTATIONS,
            pixels_per_cell=PIXELS_PER_CELL,
            cells_per_block=CELLS_PER_BLOCK,
            block_norm='L2-Hys',
            feature_vector=True
        )
        features.append(feat)
    return np.array(features, dtype=np.float32)


print("\n开始提取 HOG 特征...")
start_time = time.time()

X_train_hog = extract_hog_features(X_train_gray)
X_test_hog = extract_hog_features(X_test_gray)

print("HOG训练集形状：", X_train_hog.shape)
print("HOG测试集形状：", X_test_hog.shape)
print("HOG提取耗时：{:.2f} 秒".format(time.time() - start_time))


# =========================
# 5. 原始灰度像素拉平
# =========================
print("\n开始构造原始灰度像素特征...")
X_train_raw = X_train_gray.reshape(len(X_train_gray), -1)
X_test_raw = X_test_gray.reshape(len(X_test_gray), -1)

print("原始灰度训练特征形状：", X_train_raw.shape)  # 应为 (n, 1024)
print("原始灰度测试特征形状：", X_test_raw.shape)


# =========================
# 6. PCA降维
# =========================
print("\n开始对 HOG 特征做 PCA 降维...")
start_time = time.time()

pca_hog = PCA(n_components=PCA_DIM, random_state=42)
X_train_hog_pca = pca_hog.fit_transform(X_train_hog)
X_test_hog_pca = pca_hog.transform(X_test_hog)

print("HOG降维后训练特征形状：", X_train_hog_pca.shape)
print("HOG降维后测试特征形状：", X_test_hog_pca.shape)
print("HOG PCA累计解释方差比：{:.4f}".format(np.sum(pca_hog.explained_variance_ratio_)))
print("HOG PCA耗时：{:.2f} 秒".format(time.time() - start_time))

print("\n开始对原始灰度像素特征做 PCA 降维...")
start_time = time.time()

pca_raw = PCA(n_components=PCA_DIM, random_state=42)
X_train_raw_pca = pca_raw.fit_transform(X_train_raw)
X_test_raw_pca = pca_raw.transform(X_test_raw)

print("原始像素降维后训练特征形状：", X_train_raw_pca.shape)
print("原始像素降维后测试特征形状：", X_test_raw_pca.shape)
print("原始像素 PCA累计解释方差比：{:.4f}".format(np.sum(pca_raw.explained_variance_ratio_)))
print("原始像素 PCA耗时：{:.2f} 秒".format(time.time() - start_time))


# =========================
# 7. KNN分类：HOG + PCA + KNN
# =========================
print("\n开始训练 HOG + PCA + KNN...")
start_time = time.time()

knn_hog = KNeighborsClassifier(n_neighbors=K)
knn_hog.fit(X_train_hog_pca, y_train)
y_pred_hog = knn_hog.predict(X_test_hog_pca)

hog_acc = accuracy_score(y_test, y_pred_hog)

print("HOG + PCA + KNN 测试准确率：{:.4f}".format(hog_acc))
print("HOG + PCA + KNN 总耗时：{:.2f} 秒".format(time.time() - start_time))


# =========================
# 8. KNN分类：原始像素 + PCA + KNN
# =========================
print("\n开始训练 原始像素 + PCA + KNN...")
start_time = time.time()

knn_raw = KNeighborsClassifier(n_neighbors=K)
knn_raw.fit(X_train_raw_pca, y_train)
y_pred_raw = knn_raw.predict(X_test_raw_pca)

raw_acc = accuracy_score(y_test, y_pred_raw)

print("原始像素 + PCA + KNN 测试准确率：{:.4f}".format(raw_acc))
print("原始像素 + PCA + KNN 总耗时：{:.2f} 秒".format(time.time() - start_time))


# =========================
# 9. 对比结果输出
# =========================
print("\n================ 最终对比结果 ================")
print("HOG + PCA + KNN 准确率：{:.4f}".format(hog_acc))
print("原始像素 + PCA + KNN 准确率：{:.4f}".format(raw_acc))

if hog_acc > raw_acc:
    print("结论：HOG 特征效果更好。")
elif hog_acc < raw_acc:
    print("结论：原始像素特征效果更好。")
else:
    print("结论：两种方法效果相同。")


# =========================
# 10. 绘制对比柱状图
# =========================
methods = ["HOG+PCA+KNN", "原始像素+PCA+KNN"]
accs = [hog_acc, raw_acc]

plt.figure(figsize=(7, 5))
bars = plt.bar(methods, accs)
plt.ylabel("测试准确率")
plt.title("两种特征表示方法的分类效果对比")

for bar, acc in zip(bars, accs):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{acc:.4f}",
        ha='center',
        va='bottom'
    )

plt.ylim(0, max(accs) + 0.1)
plt.tight_layout()
plt.show()


# =========================
# 11. 输出分类报告
# =========================
print("\n================ HOG 方法分类报告 ================")
print(classification_report(y_test, y_pred_hog, digits=4))

print("\n================ 原始像素方法分类报告 ================")
print(classification_report(y_test, y_pred_raw, digits=4))


# =========================
# 12. 显示部分预测结果
# =========================
def show_predictions(images, y_true, y_pred, title, class_names, num=10):
    plt.figure(figsize=(12, 5))
    for i in range(num):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i])
        plt.title(f"真:{class_names[y_true[i]]}\n预测:{class_names[y_pred[i]]}")
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


show_predictions(X_test, y_test, y_pred_hog, "HOG+PCA+KNN 预测示例", class_names, num=10)
show_predictions(X_test, y_test, y_pred_raw, "原始像素+PCA+KNN 预测示例", class_names, num=10)