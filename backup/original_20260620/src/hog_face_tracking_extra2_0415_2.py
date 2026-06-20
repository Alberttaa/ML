import os
import cv2
import time
import math
import queue
import joblib
import random
import threading
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ======================================================
# 附加题2简略思路：
# 1. 使用 face 文件夹作为人脸正样本，background 文件夹作为背景负样本；
# 2. 对正负样本都做数据增广，让样本更丰富；
# 3. 对每张样本图像提取 HOG 特征；
# 4. 使用逻辑回归训练“人脸/非人脸”二分类器；
# 5. 摄像头实时检测时，使用滑动窗口 + 多尺度扫描；
# 6. 对每个窗口提取 HOG 特征，再用模型判断是不是人脸；
# 7. 对候选框做 NMS 去重，只保留最优人脸框；
# 8. 使用独立线程读取摄像头，主线程处理最新帧，减少卡顿；
# 9. 对检测框做平滑处理，让框更稳定、更完整地框住人脸。
# ======================================================


# =========================
# 1. 路径配置
# =========================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

FACE_DIR = os.path.join(PROJECT_ROOT, "data", "dataset", "face")
BG_DIR = os.path.join(PROJECT_ROOT, "data", "dataset", "background")

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "hog_face_tracking_extra2.pkl")


# =========================
# 2. 运行模式
# =========================

# 第一次运行：MODE = "train"
# 训练完成后：MODE = "camera"
MODE = "camera"

CAMERA_ID = 0

# HOG 输入尺寸
IMG_SIZE = 64

# 检测阈值，越高越严格，误检少但可能漏检
DETECT_THRESHOLD = 0.75

# 滑动窗口步长，越小越准确但越慢
STEP_SIZE = 24

# 摄像头画面大小
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# 多尺度窗口大小，适应不同距离的人脸
WINDOW_SIZES = [96, 128, 160, 192, 224]

# 是否显示检测耗时和 FPS
SHOW_FPS = True


# =========================
# 3. 创建文件夹
# =========================

def make_dirs():
    os.makedirs(FACE_DIR, exist_ok=True)
    os.makedirs(BG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)


# =========================
# 4. 支持中文路径读取图片
# =========================

def read_image(path):
    """
    cv2.imread 有时无法读取中文路径；
    所以这里使用 np.fromfile + cv2.imdecode。
    """
    try:
        img_array = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def get_image_files(folder):
    image_exts = [".jpg", ".jpeg", ".png", ".bmp"]
    files = []

    if not os.path.exists(folder):
        return files

    for filename in os.listdir(folder):
        ext = os.path.splitext(filename)[1].lower()

        if ext in image_exts:
            files.append(os.path.join(folder, filename))

    return files


# =========================
# 5. HOG 特征提取
# =========================

def create_hog():
    """
    HOG 提取图像的边缘和梯度方向信息。
    这里统一输入 64x64 图像。
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
    输入图片，输出一维 HOG 特征。
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

def augment_face_image(img):
    """
    人脸正样本增广：
    轻微旋转、亮度变化、水平翻转、中心裁剪。
    注意幅度不能太大，否则人脸会变形。
    """
    imgs = []

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    imgs.append(img)

    # 水平翻转
    imgs.append(cv2.flip(img, 1))

    # 亮度变化
    imgs.append(cv2.convertScaleAbs(img, alpha=1.2, beta=20))
    imgs.append(cv2.convertScaleAbs(img, alpha=0.8, beta=-20))

    # 小角度旋转
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    for angle in [-12, -6, 6, 12]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        imgs.append(rotated)

    # 轻微缩放裁剪
    crop = img[4:60, 4:60]
    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    imgs.append(crop)

    return imgs


def augment_bg_image(img):
    """
    背景负样本增广：
    随机裁剪多个区域，模拟房间背景、衣服、墙面等非人脸区域。
    """
    imgs = []

    h, w = img.shape[:2]

    if h < 20 or w < 20:
        return imgs

    # 原图缩放作为一个背景样本
    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    imgs.append(resized)

    # 随机裁剪多个背景块
    for _ in range(8):
        crop_size = random.randint(min(h, w) // 3, min(h, w))

        crop_size = max(20, crop_size)
        crop_size = min(crop_size, h, w)

        x1 = random.randint(0, max(0, w - crop_size))
        y1 = random.randint(0, max(0, h - crop_size))

        crop = img[y1:y1 + crop_size, x1:x1 + crop_size]

        if crop.size == 0:
            continue

        crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
        imgs.append(crop)

    # 亮度变化
    more_imgs = []
    for one in imgs:
        more_imgs.append(cv2.convertScaleAbs(one, alpha=1.2, beta=20))
        more_imgs.append(cv2.convertScaleAbs(one, alpha=0.8, beta=-20))

    imgs.extend(more_imgs)

    return imgs


# =========================
# 7. 加载数据集
# =========================

def load_dataset():
    """
    从 face 和 background 文件夹读取图片，
    增广后提取 HOG 特征。
    """
    hog = create_hog()

    X = []
    y = []

    face_files = get_image_files(FACE_DIR)
    bg_files = get_image_files(BG_DIR)

    print("人脸正样本目录：", FACE_DIR)
    print("背景负样本目录：", BG_DIR)
    print("原始人脸图片数量：", len(face_files))
    print("原始背景图片数量：", len(bg_files))

    if len(face_files) == 0:
        print("警告：face 文件夹没有图片。")

    if len(bg_files) == 0:
        print("警告：background 文件夹没有图片。")

    # 正样本：人脸，标签 1
    for path in face_files:
        img = read_image(path)

        if img is None:
            print("读取失败：", path)
            continue

        for aug in augment_face_image(img):
            feature = extract_hog_feature(aug, hog)
            if feature is not None:
                X.append(feature)
                y.append(1)

    # 负样本：背景，标签 0
    for path in bg_files:
        img = read_image(path)

        if img is None:
            print("读取失败：", path)
            continue

        for aug in augment_bg_image(img):
            feature = extract_hog_feature(aug, hog)
            if feature is not None:
                X.append(feature)
                y.append(0)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print("增广后 X 维度：", X.shape)
    print("增广后 y 维度：", y.shape)

    if len(y) > 0:
        print("增广后人脸样本数量：", np.sum(y == 1))
        print("增广后背景样本数量：", np.sum(y == 0))

    return X, y


# =========================
# 8. 训练模型
# =========================

def train_model():
    """
    使用 HOG 特征 + 逻辑回归训练二分类模型。
    """
    X, y = load_dataset()

    if len(X) == 0:
        print("没有读到训练数据，请检查 face 和 background 文件夹。")
        return

    if len(np.unique(y)) < 2:
        print("训练失败：必须同时有人脸正样本和背景负样本。")
        return

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced"
        ))
    ])

    print("\n开始训练 HOG + Logistic Regression 模型...")
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)

    print("\n========== 训练结果 ==========")
    print("训练集准确率：", train_acc)
    print("验证集准确率：", val_acc)

    print("\n验证集分类报告：")
    print(classification_report(y_val, val_pred, target_names=["background", "face"]))

    joblib.dump(model, MODEL_PATH)

    print("\n模型已保存到：")
    print(MODEL_PATH)

    plt.figure(figsize=(5, 4))
    plt.bar(["Train Acc", "Val Acc"], [train_acc, val_acc])
    plt.ylim(0, 1)
    plt.title("HOG Face Classifier Accuracy")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.show()


# =========================
# 9. 摄像头读取线程
# =========================

class CameraThread:
    """
    独立线程读取摄像头。
    主线程只处理最新一帧，避免摄像头缓冲导致画面延迟。
    """

    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        self.frame_queue = queue.Queue(maxsize=1)
        self.running = False
        self.thread = None

    def start(self):
        if not self.cap.isOpened():
            print("摄像头打开失败，可以把 CAMERA_ID 改成 1。")
            return False

        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

        return True

    def update(self):
        while self.running:
            ret, frame = self.cap.read()

            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # 队列满时丢掉旧帧，只保留最新帧
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass

            self.frame_queue.put(frame)

    def read(self):
        try:
            frame = self.frame_queue.get(timeout=1)
            return frame
        except queue.Empty:
            return None

    def stop(self):
        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=1)

        self.cap.release()


# =========================
# 10. 滑动窗口检测
# =========================

def sliding_windows(frame):
    """
    多尺度滑动窗口。
    窗口尺寸越多越准确，但速度会变慢。
    """
    h, w = frame.shape[:2]

    for win_size in WINDOW_SIZES:
        for y in range(0, h - win_size, STEP_SIZE):
            for x in range(0, w - win_size, STEP_SIZE):
                window = frame[y:y + win_size, x:x + win_size]
                yield x, y, win_size, win_size, window


def expand_face_box(x, y, w, h, frame_w, frame_h):
    """
    为了完整框住人脸，把检测框略微扩大。
    """
    cx = x + w / 2
    cy = y + h / 2

    new_w = w * 1.15
    new_h = h * 1.15

    new_x = int(cx - new_w / 2)
    new_y = int(cy - new_h / 2)

    new_x = max(0, new_x)
    new_y = max(0, new_y)

    new_w = min(frame_w - new_x, int(new_w))
    new_h = min(frame_h - new_y, int(new_h))

    return new_x, new_y, new_w, new_h


def nms(candidates, iou_threshold=0.3):
    """
    非极大值抑制：
    多个重叠框只保留置信度最高的框。
    candidates 格式：[x, y, w, h, prob]
    """
    if len(candidates) == 0:
        return []

    boxes = []
    scores = []

    for x, y, w, h, prob in candidates:
        boxes.append([int(x), int(y), int(w), int(h)])
        scores.append(float(prob))

    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes,
        scores=scores,
        score_threshold=DETECT_THRESHOLD,
        nms_threshold=iou_threshold
    )

    if len(indices) == 0:
        return []

    indices = np.array(indices).flatten()

    result = []
    for i in indices:
        result.append(candidates[i])

    return result


def choose_best_box(candidates, frame_w, frame_h):
    """
    最终只输出唯一检测框。
    综合考虑模型置信度和是否靠近画面中心。
    """
    if len(candidates) == 0:
        return None

    frame_cx = frame_w / 2
    frame_cy = frame_h / 2

    best = None
    best_score = -1

    for x, y, w, h, prob in candidates:
        cx = x + w / 2
        cy = y + h / 2

        distance = math.sqrt((cx - frame_cx) ** 2 + (cy - frame_cy) ** 2)
        max_distance = math.sqrt(frame_cx ** 2 + frame_cy ** 2)

        center_score = 1 - distance / max_distance

        final_score = prob * 0.85 + center_score * 0.15

        if final_score > best_score:
            best_score = final_score
            best = (x, y, w, h, prob)

    return best


def smooth_box(last_box, new_box, alpha=0.65):
    """
    检测框平滑：
    避免框在连续帧中抖动。
    alpha 越大，越相信上一帧；越小，越相信当前检测。
    """
    if last_box is None:
        return new_box

    if new_box is None:
        return last_box

    lx, ly, lw, lh, lp = last_box
    nx, ny, nw, nh, npb = new_box

    x = int(alpha * lx + (1 - alpha) * nx)
    y = int(alpha * ly + (1 - alpha) * ny)
    w = int(alpha * lw + (1 - alpha) * nw)
    h = int(alpha * lh + (1 - alpha) * nh)
    p = alpha * lp + (1 - alpha) * npb

    return x, y, w, h, p


# =========================
# 11. 单帧检测
# =========================

def detect_face_in_frame(frame, model, hog):
    """
    对一帧图像做 HOG + 逻辑回归检测。
    """
    frame_h, frame_w = frame.shape[:2]

    candidates = []

    for x, y, w, h, window in sliding_windows(frame):
        feature = extract_hog_feature(window, hog)

        if feature is None:
            continue

        feature = feature.reshape(1, -1)

        prob = model.predict_proba(feature)[0][1]

        if prob >= DETECT_THRESHOLD:
            ex, ey, ew, eh = expand_face_box(x, y, w, h, frame_w, frame_h)
            candidates.append((ex, ey, ew, eh, prob))

    candidates = nms(candidates, iou_threshold=0.3)

    best_box = choose_best_box(candidates, frame_w, frame_h)

    return best_box, len(candidates)


# =========================
# 12. 摄像头实时检测
# =========================

def camera_detect():
    if not os.path.exists(MODEL_PATH):
        print("没有找到模型文件，请先设置 MODE = 'train' 训练模型。")
        print("模型路径：", MODEL_PATH)
        return

    model = joblib.load(MODEL_PATH)
    hog = create_hog()

    cam = CameraThread(CAMERA_ID)

    if not cam.start():
        return

    print("摄像头已打开，按 q 退出。")
    print("当前检测阈值：", DETECT_THRESHOLD)
    print("滑动窗口步长：", STEP_SIZE)
    print("窗口尺寸：", WINDOW_SIZES)

    last_box = None
    last_time = time.time()
    fps = 0

    while True:
        frame = cam.read()

        if frame is None:
            continue

        start_time = time.time()

        best_box, candidate_count = detect_face_in_frame(frame, model, hog)

        # 平滑检测框
        if best_box is not None:
            last_box = smooth_box(last_box, best_box, alpha=0.55)
        else:
            last_box = None

        # 绘制唯一检测框
        if last_box is not None:
            x, y, w, h, prob = last_box

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"Face {prob:.2f}",
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

        end_time = time.time()
        elapsed = end_time - start_time

        if elapsed > 0:
            fps = 1.0 / elapsed

        if SHOW_FPS:
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )

            cv2.putText(
                frame,
                f"Candidates: {candidate_count}",
                (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )

        cv2.imshow("Improved HOG Face Tracking", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cam.stop()
    cv2.destroyAllWindows()


# =========================
# 13. 主函数
# =========================

def main():
    make_dirs()

    print("项目根目录：", PROJECT_ROOT)
    print("人脸正样本目录：", FACE_DIR)
    print("背景负样本目录：", BG_DIR)
    print("模型保存路径：", MODEL_PATH)
    print("当前模式：", MODE)

    if MODE == "train":
        train_model()

    elif MODE == "camera":
        camera_detect()

    else:
        print("MODE 设置错误。")
        print("请设置 MODE = 'train' 或 MODE = 'camera'")


if __name__ == "__main__":
    main()