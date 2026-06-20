import os
import cv2
import joblib
import random
import math
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss


# ======================================================
# 附加题2简略思路：
# 1. face 文件夹放本人脸正样本，background 文件夹放背景负样本；
# 2. 对每张图片统一尺寸，然后提取 HOG 特征，组成数据 X；
# 3. 人脸标签为 1，背景标签为 0；
# 4. 使用逻辑回归训练二分类模型；
# 5. 摄像头实时读取画面，用滑动窗口扫描可能的人脸区域；
# 6. 每个窗口提取 HOG 特征，送入逻辑回归模型判断是否为人脸；
# 7. 每帧只保留置信度最高的窗口，输出唯一检测框。
# ======================================================


# =========================
# 1. 路径配置
# =========================

# 当前文件位置：My project/src/hog_face_logistic.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 项目根目录：My project
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# 数据集目录：
# My project/data/dataset/face
# My project/data/dataset/background
FACE_DIR = os.path.join(PROJECT_ROOT, "data", "dataset", "face")
BG_DIR = os.path.join(PROJECT_ROOT, "data", "dataset", "background")

# 模型保存路径
MODEL_PATH = os.path.join(PROJECT_ROOT, "hog_face_lr_model.pkl")


# =========================
# 2. 运行模式
# =========================

# 你现在 face 和 background 已经有照片，所以先用 train
# 第一次：MODE = "train"
# 训练完成后：MODE = "camera"
MODE = "camera"

CAMERA_ID = 0

# HOG 输入图像大小
IMG_SIZE = 64

# 摄像头检测阈值，越大越严格，越小越容易检测到但误检更多
DETECT_THRESHOLD = 0.75

# 滑动窗口步长，越小越准但越慢
STEP_SIZE = 16


# =========================
# 3. 创建文件夹
# =========================

def make_dirs():
    os.makedirs(FACE_DIR, exist_ok=True)
    os.makedirs(BG_DIR, exist_ok=True)


# =========================
# 4. 支持中文路径读取图片
# =========================

def read_image(path):
    """
    cv2.imread 有时读不了中文路径；
    这里用 np.fromfile + cv2.imdecode，更稳一点。
    """
    if not os.path.exists(path):
        return None

    try:
        img_array = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


# =========================
# 5. HOG 特征提取
# =========================

def create_hog():
    """
    HOG 用来提取图像边缘和梯度方向特征。
    每张图片统一为 64x64，再转成 HOG 特征向量。
    """
    hog = cv2.HOGDescriptor(
        _winSize=(IMG_SIZE, IMG_SIZE),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )
    return hog


def extract_hog_feature(img, hog):
    """
    输入图片，输出一维 HOG 特征向量。
    """
    if img is None:
        return None

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    feature = hog.compute(gray)
    feature = feature.flatten()

    return feature


# =========================
# 6. 数据增广
# =========================

def augment_image(img):
    """
    数据增广：
    因为每类样本大概只有10张，所以通过翻转、亮度变化、旋转扩充数据。
    """
    imgs = []

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    imgs.append(img)

    # 水平翻转
    imgs.append(cv2.flip(img, 1))

    # 亮一点
    brighter = cv2.convertScaleAbs(img, alpha=1.2, beta=20)
    imgs.append(brighter)

    # 暗一点
    darker = cv2.convertScaleAbs(img, alpha=0.8, beta=-20)
    imgs.append(darker)

    # 小角度旋转
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    for angle in [-10, 10]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        imgs.append(rotated)

    return imgs


# =========================
# 7. 加载数据集
# =========================

def get_image_files(folder):
    """
    读取文件夹中的图片文件。
    """
    image_exts = [".jpg", ".jpeg", ".png", ".bmp"]

    files = []

    for filename in os.listdir(folder):
        ext = os.path.splitext(filename)[1].lower()
        if ext in image_exts:
            files.append(os.path.join(folder, filename))

    return files


def load_dataset():
    """
    读取 face 和 background 文件夹图片，
    提取 HOG 特征，形成 X 和 y。
    """
    hog = create_hog()

    X = []
    y = []

    face_files = get_image_files(FACE_DIR)
    bg_files = get_image_files(BG_DIR)

    print("人脸正样本路径：", FACE_DIR)
    print("背景负样本路径：", BG_DIR)
    print("原始人脸图片数量：", len(face_files))
    print("原始背景图片数量：", len(bg_files))

    if len(face_files) == 0:
        print("警告：face 文件夹中没有读到图片。")

    if len(bg_files) == 0:
        print("警告：background 文件夹中没有读到图片。")

    # 读取人脸正样本，标签为 1
    for path in face_files:
        img = read_image(path)

        if img is None:
            print("读取失败：", path)
            continue

        for aug_img in augment_image(img):
            feature = extract_hog_feature(aug_img, hog)
            if feature is not None:
                X.append(feature)
                y.append(1)

    # 读取背景负样本，标签为 0
    for path in bg_files:
        img = read_image(path)

        if img is None:
            print("读取失败：", path)
            continue

        for aug_img in augment_image(img):
            feature = extract_hog_feature(aug_img, hog)
            if feature is not None:
                X.append(feature)
                y.append(0)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print("增广后数据维度 X：", X.shape)
    print("标签维度 y：", y.shape)

    if len(y) > 0:
        print("增广后人脸样本数量：", np.sum(y == 1))
        print("增广后背景样本数量：", np.sum(y == 0))

    return X, y


# =========================
# 8. 训练逻辑回归模型
# =========================

def train_model():
    """
    训练 HOG + 逻辑回归模型。
    """
    X, y = load_dataset()

    if len(X) == 0:
        print("没有读到任何训练数据，请检查 data/dataset/face 和 background 文件夹。")
        return

    if len(np.unique(y)) < 2:
        print("训练失败：必须同时有人脸正样本和背景负样本。")
        return

    # Pipeline：先标准化，再逻辑回归
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, C=1.0))
    ])

    model.fit(X, y)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    train_acc = accuracy_score(y, y_pred)
    train_loss = log_loss(y, y_prob)

    print("\n========== 训练结果 ==========")
    print("训练集准确率：", train_acc)
    print("训练集 log loss：", train_loss)

    joblib.dump(model, MODEL_PATH)

    print("\n模型已保存到：")
    print(MODEL_PATH)

    # 简单画一个训练结果柱状图
    plt.figure(figsize=(5, 4))
    plt.bar(["Train Accuracy"], [train_acc])
    plt.ylim(0, 1)
    plt.title("HOG + Logistic Regression Training Accuracy")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.show()


# =========================
# 9. 滑动窗口扫描
# =========================

def sliding_windows(frame):
    """
    在摄像头画面中用不同大小的窗口扫描。
    每个窗口都可能是人脸候选区域。
    """
    h, w = frame.shape[:2]

    # 窗口大小可以根据摄像头距离调整
    window_sizes = [96, 128, 160, 192, 224]

    for win_size in window_sizes:
        for y in range(0, h - win_size, STEP_SIZE):
            for x in range(0, w - win_size, STEP_SIZE):
                window = frame[y:y + win_size, x:x + win_size]
                yield x, y, win_size, win_size, window


# =========================
# 10. 选择最优检测框
# =========================

def choose_best_box(candidates, frame_w, frame_h):
    """
    如果多个窗口都像人脸，只选择一个最优框。
    综合考虑：
    1. 模型置信度；
    2. 是否靠近画面中心。
    """
    if len(candidates) == 0:
        return None

    frame_cx = frame_w / 2
    frame_cy = frame_h / 2

    best_candidate = None
    best_score = -1

    for box in candidates:
        x, y, w, h, prob = box

        cx = x + w / 2
        cy = y + h / 2

        distance = math.sqrt((cx - frame_cx) ** 2 + (cy - frame_cy) ** 2)
        max_distance = math.sqrt(frame_cx ** 2 + frame_cy ** 2)

        center_score = 1 - distance / max_distance

        # 最终得分：置信度为主，中心位置为辅
        final_score = prob * 0.8 + center_score * 0.2

        if final_score > best_score:
            best_score = final_score
            best_candidate = box

    return best_candidate


# =========================
# 11. 摄像头实时检测
# =========================

def camera_detect():
    """
    打开摄像头，使用训练好的模型进行人脸实时检测。
    每一帧只输出唯一的检测框。
    """
    if not os.path.exists(MODEL_PATH):
        print("没有找到模型文件，请先把 MODE 改成 train 训练模型。")
        print("当前模型路径：", MODEL_PATH)
        return

    model = joblib.load(MODEL_PATH)
    hog = create_hog()

    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        print("摄像头打开失败。如果你有多个摄像头，可以把 CAMERA_ID 改成 1。")
        return

    print("摄像头已打开，按 q 退出。")
    print("当前检测阈值 DETECT_THRESHOLD =", DETECT_THRESHOLD)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("摄像头读取失败。")
            break

        # 镜像一下，更符合前置摄像头习惯
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))

        frame_h, frame_w = frame.shape[:2]

        candidates = []

        # 滑动窗口扫描整张图
        for x, y, w, h, window in sliding_windows(frame):
            feature = extract_hog_feature(window, hog)

            if feature is None:
                continue

            feature = feature.reshape(1, -1)

            # 属于人脸的概率
            prob = model.predict_proba(feature)[0][1]

            if prob >= DETECT_THRESHOLD:
                candidates.append((x, y, w, h, prob))

        best_box = choose_best_box(candidates, frame_w, frame_h)

        # 只画一个检测框
        if best_box is not None:
            x, y, w, h, prob = best_box

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"Face: {prob:.2f}",
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
        else:
            cv2.putText(
                frame,
                "No face detected",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.imshow("HOG + Logistic Regression Face Tracking", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# =========================
# 12. 主函数
# =========================

def main():
    make_dirs()

    print("项目根目录：", PROJECT_ROOT)
    print("人脸样本目录：", FACE_DIR)
    print("背景样本目录：", BG_DIR)
    print("模型保存路径：", MODEL_PATH)
    print("当前运行模式：", MODE)

    if MODE == "train":
        train_model()

    elif MODE == "camera":
        camera_detect()

    else:
        print("MODE 设置错误。")
        print('请设置为 MODE = "train" 或 MODE = "camera"')


if __name__ == "__main__":
    main()