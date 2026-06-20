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
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ======================================================
# 附加题2简略思路：
# 1. face 文件夹放本人脸正样本，background 文件夹放背景负样本；
# 2. 对正负样本做数据增广，让样本更丰富；
# 3. 每张图片统一为 64x64，并提取 HOG 特征；
# 4. 分别训练 SVM 和逻辑回归二分类器；
# 5. 在验证集上比较二者准确率和预测速度；
# 6. 摄像头实时检测时，用多尺度滑动窗口扫描画面；
# 7. 每个窗口提取 HOG 特征，再交给 SVM 判断是否为人脸；
# 8. 对候选框做 NMS 去重，只保留一个最优框；
# 9. 使用摄像头读取线程，减少卡顿，提高实时性。
# ======================================================


# =========================
# 1. 路径配置
# =========================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

FACE_DIR = os.path.join(PROJECT_ROOT, "data", "dataset", "face")
BG_DIR = os.path.join(PROJECT_ROOT, "data", "dataset", "background")

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

SVM_MODEL_PATH = os.path.join(MODEL_DIR, "hog_face_svm.pkl")
LR_MODEL_PATH = os.path.join(MODEL_DIR, "hog_face_logistic.pkl")


# =========================
# 2. 运行模式
# =========================

# 第一次运行：MODE = "train_compare"
# 训练后用 SVM 摄像头检测：MODE = "camera_svm"
# 训练后用逻辑回归摄像头检测：MODE = "camera_lr"
MODE = "camera_lr"

CAMERA_ID = 0

# HOG 输入尺寸
IMG_SIZE = 64

# 摄像头画面大小
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# 滑动窗口参数
# STEP_SIZE 越小框越准，但越慢；越大越快，但可能漏检
STEP_SIZE = 32

# 多尺度窗口，适应远近不同的人脸
WINDOW_SIZES = [96, 128, 160, 192]

# SVM 阈值：decision_function 大于该值认为是人脸
SVM_THRESHOLD = 0.0

# 逻辑回归阈值：人脸概率大于该值认为是人脸
LR_THRESHOLD = 0.75

# 是否显示 FPS
SHOW_FPS = True


# =========================
# 3. 创建文件夹
# =========================

def make_dirs():
    os.makedirs(FACE_DIR, exist_ok=True)
    os.makedirs(BG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================
# 4. 读取图片
# =========================

def read_image(path):
    """
    支持中文路径读取图片。
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
    HOG 用来提取图像的边缘、纹理和梯度方向特征。
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
    输入一张图像，输出一维 HOG 特征。
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
    翻转、亮度变化、轻微旋转、轻微裁剪。
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

    # 轻微中心裁剪再放大
    crop = img[4:60, 4:60]
    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    imgs.append(crop)

    return imgs


def augment_background_image(img):
    """
    背景负样本增广：
    随机裁剪背景块，让负样本覆盖桌面、衣服、墙、手等非人脸区域。
    """
    imgs = []

    h, w = img.shape[:2]

    if h < 20 or w < 20:
        return imgs

    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    imgs.append(resized)

    # 随机裁剪多个背景块
    for _ in range(10):
        crop_size = random.randint(max(20, min(h, w) // 4), min(h, w))

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
    从 face 和 background 读取图片，增广后提取 HOG 特征。
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

        for aug in augment_background_image(img):
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
# 8. 模型训练与比较
# =========================

def build_svm_model():
    """
    使用线性 SVM。
    HOG 特征通常适合配合线性 SVM，速度也比 RBF SVM 快。
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(
            C=1.0,
            class_weight="balanced",
            max_iter=5000
        ))
    ])

    return model


def build_logistic_model():
    """
    逻辑回归作为对比模型。
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    return model


def benchmark_predict_speed(model, X_val, repeat=50):
    """
    比较模型预测速度。
    只比较模型预测，不包括摄像头读取和 HOG 提取时间。
    """
    sample = X_val[:min(200, len(X_val))]

    start_time = time.time()

    for _ in range(repeat):
        model.predict(sample)

    end_time = time.time()

    total_samples = len(sample) * repeat
    avg_time = (end_time - start_time) / total_samples

    return avg_time


def train_and_compare():
    """
    训练 SVM 和逻辑回归，并比较准确率和速度。
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

    models = {
        "SVM": build_svm_model(),
        "Logistic Regression": build_logistic_model()
    }

    results = {}

    for name, model in models.items():
        print("\n" + "=" * 60)
        print("开始训练模型：", name)
        print("=" * 60)

        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)

        avg_time = benchmark_predict_speed(model, X_val)

        print(f"{name} 训练集准确率：{train_acc:.4f}")
        print(f"{name} 验证集准确率：{val_acc:.4f}")
        print(f"{name} 平均单样本预测时间：{avg_time * 1000:.6f} ms")

        print("\n验证集分类报告：")
        print(classification_report(y_val, val_pred, target_names=["background", "face"]))

        results[name] = {
            "model": model,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "avg_time": avg_time
        }

    joblib.dump(results["SVM"]["model"], SVM_MODEL_PATH)
    joblib.dump(results["Logistic Regression"]["model"], LR_MODEL_PATH)

    print("\n模型保存完成：")
    print("SVM 模型：", SVM_MODEL_PATH)
    print("逻辑回归模型：", LR_MODEL_PATH)

    save_compare_result(results)
    plot_compare_result(results)


def save_compare_result(results):
    """
    保存准确率和速度对比结果。
    """
    save_path = os.path.join(RESULTS_DIR, "svm_lr_face_tracking_compare.txt")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("HOG 人脸跟踪：SVM 与逻辑回归对比结果\n")
        f.write("=" * 50 + "\n\n")

        for name, info in results.items():
            f.write(f"模型：{name}\n")
            f.write(f"训练集准确率：{info['train_acc']:.4f}\n")
            f.write(f"验证集准确率：{info['val_acc']:.4f}\n")
            f.write(f"平均单样本预测时间：{info['avg_time'] * 1000:.6f} ms\n\n")

    print("对比结果已保存：", save_path)


def plot_compare_result(results):
    """
    画准确率和速度对比图。
    """
    names = list(results.keys())
    val_accs = [results[name]["val_acc"] for name in names]
    times = [results[name]["avg_time"] * 1000 for name in names]

    plt.figure(figsize=(7, 5))
    plt.bar(names, val_accs)
    plt.ylim(0, 1)
    plt.ylabel("Validation Accuracy")
    plt.title("SVM vs Logistic Regression Accuracy")

    for i, acc in enumerate(val_accs):
        plt.text(i, acc + 0.02, f"{acc:.4f}", ha="center")

    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, "svm_lr_accuracy_compare.png")
    plt.savefig(save_path, dpi=300)
    print("准确率对比图已保存：", save_path)
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.bar(names, times)
    plt.ylabel("Avg Prediction Time per Sample (ms)")
    plt.title("SVM vs Logistic Regression Speed")

    for i, t in enumerate(times):
        plt.text(i, t + 0.001, f"{t:.4f}", ha="center")

    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, "svm_lr_speed_compare.png")
    plt.savefig(save_path, dpi=300)
    print("速度对比图已保存：", save_path)
    plt.show()


# =========================
# 9. 摄像头线程
# =========================

class CameraThread:
    """
    独立线程读取摄像头。
    Windows 下使用 CAP_DSHOW，避免 MSMF 后端取帧失败。
    """

    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        self.frame_queue = queue.Queue(maxsize=1)
        self.running = False
        self.thread = None

    def start(self):
        if not self.cap.isOpened():
            print("摄像头打开失败。")
            print("可以尝试把 CAMERA_ID 改成 1 或 2。")
            return False

        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        return True

    def update(self):
        fail_count = 0

        while self.running:
            ret, frame = self.cap.read()

            if not ret or frame is None:
                fail_count += 1
                time.sleep(0.03)

                if fail_count % 30 == 0:
                    print("摄像头暂时没有读取到画面，请检查是否被占用。")

                continue

            fail_count = 0

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

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
# 10. 摄像头检测相关函数
# =========================

def sliding_windows(frame):
    """
    多尺度滑动窗口扫描图像。
    """
    h, w = frame.shape[:2]

    for win_size in WINDOW_SIZES:
        for y in range(0, h - win_size, STEP_SIZE):
            for x in range(0, w - win_size, STEP_SIZE):
                window = frame[y:y + win_size, x:x + win_size]
                yield x, y, win_size, win_size, window


def expand_face_box(x, y, w, h, frame_w, frame_h):
    """
    为了完整框住人脸，将检测框略微扩大。
    """
    cx = x + w / 2
    cy = y + h / 2

    new_w = w * 1.18
    new_h = h * 1.18

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
    多个重叠框只保留分数最高的框。
    candidates: [x, y, w, h, score]
    """
    if len(candidates) == 0:
        return []

    boxes = []
    scores = []

    for x, y, w, h, score in candidates:
        boxes.append([int(x), int(y), int(w), int(h)])
        scores.append(float(score))

    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes,
        scores=scores,
        score_threshold=0.0,
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
    每一帧只输出唯一检测框。
    综合考虑模型分数和是否靠近画面中心。
    """
    if len(candidates) == 0:
        return None

    frame_cx = frame_w / 2
    frame_cy = frame_h / 2

    best = None
    best_score = -1e9

    for x, y, w, h, score in candidates:
        cx = x + w / 2
        cy = y + h / 2

        distance = math.sqrt((cx - frame_cx) ** 2 + (cy - frame_cy) ** 2)
        max_distance = math.sqrt(frame_cx ** 2 + frame_cy ** 2)
        center_score = 1 - distance / max_distance

        final_score = score * 0.85 + center_score * 0.15

        if final_score > best_score:
            best_score = final_score
            best = (x, y, w, h, score)

    return best


def smooth_box(last_box, new_box, alpha=0.55):
    """
    平滑检测框，减少抖动。
    """
    if last_box is None:
        return new_box

    if new_box is None:
        return None

    lx, ly, lw, lh, ls = last_box
    nx, ny, nw, nh, ns = new_box

    x = int(alpha * lx + (1 - alpha) * nx)
    y = int(alpha * ly + (1 - alpha) * ny)
    w = int(alpha * lw + (1 - alpha) * nw)
    h = int(alpha * lh + (1 - alpha) * nh)
    score = alpha * ls + (1 - alpha) * ns

    return x, y, w, h, score


def get_model_score(model, feature, model_type):
    """
    根据模型类型返回人脸分数：
    - SVM 使用 decision_function；
    - 逻辑回归使用 predict_proba。
    """
    feature = feature.reshape(1, -1)

    if model_type == "svm":
        score = model.decision_function(feature)[0]
        return score

    elif model_type == "lr":
        prob = model.predict_proba(feature)[0][1]
        return prob

    else:
        raise ValueError("model_type 必须是 svm 或 lr")


def detect_face_in_frame(frame, model, model_type, hog):
    """
    对单帧画面进行检测。
    """
    frame_h, frame_w = frame.shape[:2]

    candidates = []

    for x, y, w, h, window in sliding_windows(frame):
        feature = extract_hog_feature(window, hog)

        if feature is None:
            continue

        score = get_model_score(model, feature, model_type)

        if model_type == "svm":
            is_face = score >= SVM_THRESHOLD
        else:
            is_face = score >= LR_THRESHOLD

        if is_face:
            ex, ey, ew, eh = expand_face_box(x, y, w, h, frame_w, frame_h)
            candidates.append((ex, ey, ew, eh, float(score)))

    candidates = nms(candidates, iou_threshold=0.3)
    best_box = choose_best_box(candidates, frame_w, frame_h)

    return best_box, len(candidates)


# =========================
# 11. 摄像头实时检测
# =========================

def camera_detect(model_type="svm"):
    """
    model_type:
    - svm：使用 SVM 检测
    - lr：使用逻辑回归检测
    """
    if model_type == "svm":
        model_path = SVM_MODEL_PATH
        window_name = "HOG + SVM Face Tracking"
    else:
        model_path = LR_MODEL_PATH
        window_name = "HOG + Logistic Regression Face Tracking"

    if not os.path.exists(model_path):
        print("没有找到模型文件，请先运行 MODE = 'train_compare'。")
        print("当前模型路径：", model_path)
        return

    model = joblib.load(model_path)
    hog = create_hog()

    cam = CameraThread(CAMERA_ID)

    if not cam.start():
        return

    print("摄像头已打开，按 q 退出。")
    print("当前模型：", model_type)
    print("窗口尺寸：", WINDOW_SIZES)
    print("滑动步长：", STEP_SIZE)

    last_box = None
    fps = 0

    while True:
        frame = cam.read()

        if frame is None:
            continue

        start_time = time.time()

        best_box, candidate_count = detect_face_in_frame(
            frame=frame,
            model=model,
            model_type=model_type,
            hog=hog
        )

        last_box = smooth_box(last_box, best_box, alpha=0.55)

        if last_box is not None:
            x, y, w, h, score = last_box

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"{model_type.upper()} Face {score:.2f}",
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
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

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cam.stop()
    cv2.destroyAllWindows()


# =========================
# 12. 主函数
# =========================

def main():
    make_dirs()

    print("项目根目录：", PROJECT_ROOT)
    print("人脸正样本目录：", FACE_DIR)
    print("背景负样本目录：", BG_DIR)
    print("SVM 模型路径：", SVM_MODEL_PATH)
    print("逻辑回归模型路径：", LR_MODEL_PATH)
    print("当前模式：", MODE)

    if MODE == "train_compare":
        train_and_compare()

    elif MODE == "camera_svm":
        camera_detect(model_type="svm")

    elif MODE == "camera_lr":
        camera_detect(model_type="lr")

    else:
        print("MODE 设置错误。")
        print('可选：MODE = "train_compare" / "camera_svm" / "camera_lr"')


if __name__ == "__main__":
    main()