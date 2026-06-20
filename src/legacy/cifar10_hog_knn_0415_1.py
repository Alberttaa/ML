import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score


# ======================================================
# 附加题1简略思路：
# 1. 从 data/cifar10/train 和 data/cifar10/test 中读取图片；
# 2. 每个类别文件夹名就是类别名，比如 airplane、cat、truck；
# 3. 方法一：不用 HOG，把图片转灰度并缩放成 32x32，再拉平成 1x1024；
# 4. 方法二：提取每张图片的 HOG 特征；
# 5. 因为特征维度较高，KNN 距离计算会慢，所以先标准化，再用 PCA 降维；
# 6. 手写 KNN，k=5，在测试集上分类；
# 7. 输出 Raw Pixel + KNN 和 HOG + KNN 的测试准确率并画图对比。
# ======================================================


# =========================
# 1. 路径与参数配置
# =========================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# 你的数据路径：
# My project/data/cifar10/train
# My project/data/cifar10/test
CIFAR_DIR = os.path.join(PROJECT_ROOT, "data", "cifar10")
TRAIN_DIR = os.path.join(CIFAR_DIR, "train")
TEST_DIR = os.path.join(CIFAR_DIR, "test")

# KNN 参数
K = 5

# 为了普通电脑运行快一点，可以限制数据量
# None 表示使用全部数据
TRAIN_LIMIT = 10000
TEST_LIMIT = 2000

# PCA 降维维度
RAW_PCA_DIM = 100
HOG_PCA_DIM = 80

# KNN 分批预测，避免内存过大
BATCH_SIZE = 200

# CIFAR-10 标准类别顺序
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CLASS_TO_LABEL = {name: idx for idx, name in enumerate(CLASS_NAMES)}


# =========================
# 2. 支持中文路径读取图片
# =========================

def read_image(path):
    """
    cv2.imread 有时读不了中文路径；
    用 np.fromfile + cv2.imdecode 更稳。
    """
    try:
        img_array = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


# =========================
# 3. 从文件夹读取 CIFAR-10 图片
# =========================

def load_images_from_folder(root_dir, limit=None):
    """
    root_dir 结构示例：
    train/
        airplane/
        automobile/
        bird/
        ...
    test/
        airplane/
        automobile/
        bird/
        ...

    返回：
    images: [N, 32, 32, 3]
    labels: [N]
    """

    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"找不到文件夹：{root_dir}")

    image_exts = [".jpg", ".jpeg", ".png", ".bmp"]
    images = []
    labels = []

    for class_name in CLASS_NAMES:
        class_dir = os.path.join(root_dir, class_name)

        if not os.path.exists(class_dir):
            print(f"警告：没有找到类别文件夹：{class_dir}")
            continue

        label = CLASS_TO_LABEL[class_name]

        filenames = os.listdir(class_dir)

        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()

            if ext not in image_exts:
                continue

            img_path = os.path.join(class_dir, filename)
            img = read_image(img_path)

            if img is None:
                print("图片读取失败：", img_path)
                continue

            # 统一缩放成 32x32
            img = cv2.resize(img, (32, 32))

            # OpenCV 默认是 BGR，这里转成 RGB，后面处理更统一
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            images.append(img)
            labels.append(label)

            if limit is not None and len(images) >= limit:
                return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.int32)

    return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.int32)


def load_cifar10_folder():
    """
    读取当前这种 train/test 分类文件夹格式的数据。
    """

    print("训练集路径：", TRAIN_DIR)
    print("测试集路径：", TEST_DIR)

    x_train, y_train = load_images_from_folder(TRAIN_DIR, limit=TRAIN_LIMIT)
    x_test, y_test = load_images_from_folder(TEST_DIR, limit=TEST_LIMIT)

    print("读取训练集：", x_train.shape, y_train.shape)
    print("读取测试集：", x_test.shape, y_test.shape)

    return x_train, y_train, x_test, y_test


# =========================
# 4. 原始像素特征：32x32 -> 1x1024
# =========================

def extract_raw_gray_features(images):
    """
    不用 HOG：
    将彩色图像转灰度图，再拉平成 1x1024。
    """

    features = []

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = gray.astype(np.float32) / 255.0
        feature = gray.flatten()
        features.append(feature)

    return np.array(features, dtype=np.float32)


# =========================
# 5. HOG 特征提取
# =========================

def create_hog():
    """
    HOG 提取边缘和梯度方向特征。
    对 32x32 图像使用较小窗口。
    """
    hog = cv2.HOGDescriptor(
        _winSize=(32, 32),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )
    return hog


def extract_hog_features(images):
    """
    对每一张图提取 HOG 特征。
    """

    hog = create_hog()
    features = []

    for idx, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (32, 32))

        feature = hog.compute(gray)
        feature = feature.flatten()
        features.append(feature)

        if (idx + 1) % 1000 == 0:
            print(f"HOG 特征提取进度：{idx + 1}/{len(images)}")

    return np.array(features, dtype=np.float32)


# =========================
# 6. 标准化 + PCA 降维
# =========================

def scale_and_pca(x_train, x_test, pca_dim):
    """
    KNN 依赖距离计算，特征尺度会直接影响距离大小。
    所以先标准化，再 PCA 降维。
    """

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    if pca_dim is not None and pca_dim < x_train_scaled.shape[1]:
        pca = PCA(n_components=pca_dim, random_state=42)
        x_train_pca = pca.fit_transform(x_train_scaled)
        x_test_pca = pca.transform(x_test_scaled)

        print(f"PCA 降维：{x_train.shape[1]} -> {pca_dim}")
        print(f"PCA 保留信息比例：{np.sum(pca.explained_variance_ratio_):.4f}")

        return x_train_pca.astype(np.float32), x_test_pca.astype(np.float32)

    return x_train_scaled.astype(np.float32), x_test_scaled.astype(np.float32)


# =========================
# 7. 手写 KNN，向量化计算距离
# =========================

def knn_predict(x_train, y_train, x_test, k=5, batch_size=200):
    """
    手写 KNN：
    1. 计算测试样本到所有训练样本的距离；
    2. 找最近的 k 个样本；
    3. 对 k 个标签投票；
    4. 票数最多的类别作为预测结果。

    距离计算使用向量化：
    dist(x, y)^2 = x^2 + y^2 - 2xy
    """

    y_pred = []

    train_square = np.sum(x_train ** 2, axis=1)

    for start in range(0, len(x_test), batch_size):
        end = min(start + batch_size, len(x_test))
        x_batch = x_test[start:end]

        test_square = np.sum(x_batch ** 2, axis=1, keepdims=True)

        dists = test_square + train_square.reshape(1, -1) - 2 * np.dot(x_batch, x_train.T)
        dists = np.maximum(dists, 0)

        nearest_indices = np.argpartition(dists, kth=k, axis=1)[:, :k]

        for i in range(nearest_indices.shape[0]):
            nearest_labels = y_train[nearest_indices[i]]
            counts = np.bincount(nearest_labels, minlength=10)
            pred_label = np.argmax(counts)
            y_pred.append(pred_label)

        print(f"KNN 预测进度：{end}/{len(x_test)}")

    return np.array(y_pred, dtype=np.int32)


# =========================
# 8. 单个实验流程
# =========================

def run_knn_experiment(feature_name, x_train_feat, x_test_feat, y_train, y_test, pca_dim):
    print("\n" + "=" * 60)
    print(f"开始实验：{feature_name}")
    print("=" * 60)

    print("原始特征维度：", x_train_feat.shape)

    x_train_final, x_test_final = scale_and_pca(
        x_train_feat,
        x_test_feat,
        pca_dim=pca_dim
    )

    print("处理后训练特征维度：", x_train_final.shape)
    print("处理后测试特征维度：", x_test_final.shape)

    start_time = time.time()

    y_pred = knn_predict(
        x_train_final,
        y_train,
        x_test_final,
        k=K,
        batch_size=BATCH_SIZE
    )

    end_time = time.time()

    acc = accuracy_score(y_test, y_pred)

    print(f"\n{feature_name} + KNN 测试准确率：{acc:.4f}")
    print(f"运行时间：{end_time - start_time:.2f} 秒")

    return acc


# =========================
# 9. 画准确率对比图
# =========================

def plot_result(raw_acc, hog_acc):
    names = ["Raw 32x32 Gray", "HOG"]
    accs = [raw_acc, hog_acc]

    plt.figure(figsize=(7, 5))
    plt.bar(names, accs)
    plt.ylim(0, 1)
    plt.ylabel("Test Accuracy")
    plt.title("CIFAR-10 KNN Accuracy Comparison, k=5")

    for i, acc in enumerate(accs):
        plt.text(i, acc + 0.02, f"{acc:.4f}", ha="center")

    plt.tight_layout()
    plt.show()


# =========================
# 10. 主函数
# =========================

def main():
    print("CIFAR-10 根目录：", CIFAR_DIR)
    print("类别顺序：", CLASS_NAMES)

    x_train, y_train, x_test, y_test = load_cifar10_folder()

    if len(x_train) == 0 or len(x_test) == 0:
        print("没有读到训练集或测试集图片，请检查路径和文件夹结构。")
        return

    print("\n========== 数据读取完成 ==========")
    print("训练集：", x_train.shape, y_train.shape)
    print("测试集：", x_test.shape, y_test.shape)

    # 方法一：原始灰度像素特征
    print("\n正在提取原始像素特征...")
    x_train_raw = extract_raw_gray_features(x_train)
    x_test_raw = extract_raw_gray_features(x_test)

    raw_acc = run_knn_experiment(
        feature_name="Raw Pixel",
        x_train_feat=x_train_raw,
        x_test_feat=x_test_raw,
        y_train=y_train,
        y_test=y_test,
        pca_dim=RAW_PCA_DIM
    )

    # 方法二：HOG 特征
    print("\n正在提取 HOG 特征...")
    x_train_hog = extract_hog_features(x_train)
    x_test_hog = extract_hog_features(x_test)

    hog_acc = run_knn_experiment(
        feature_name="HOG",
        x_train_feat=x_train_hog,
        x_test_feat=x_test_hog,
        y_train=y_train,
        y_test=y_test,
        pca_dim=HOG_PCA_DIM
    )

    print("\n========== 最终对比结果 ==========")
    print(f"Raw Pixel + KNN, k={K}, 测试准确率：{raw_acc:.4f}")
    print(f"HOG + KNN,       k={K}, 测试准确率：{hog_acc:.4f}")

    plot_result(raw_acc, hog_acc)


if __name__ == "__main__":
    main()