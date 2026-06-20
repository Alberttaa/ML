import cv2
import numpy as np
import os
import math


# =========================
# 1. 配置区
# =========================

CAMERA_ID = 0

# 遮挡图片路径，没有图片也没事，会自动生成默认 MASK 图片
MASK_IMAGE_PATH = r"E:\Ling\homework\aaa\data_titanic\mask.png"

# 建议只遮挡一个目标，不要遮挡所有检测框
COVER_ALL_PEOPLE = False

# 是否显示原始检测框，调试时可以设为 True
SHOW_RAW_BOXES = False

# 按 q 退出
QUIT_KEY = ord("q")


# =========================
# 2. 读取图片
# =========================

def read_image_chinese_path(path, flags=cv2.IMREAD_UNCHANGED):
    if not os.path.exists(path):
        return None

    img_array = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(img_array, flags)
    return img


def create_default_mask(width=300, height=300):
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    mask[:] = (50, 50, 50)

    cv2.rectangle(mask, (10, 10), (width - 10, height - 10), (255, 255, 255), 5)
    cv2.putText(
        mask,
        "MASK",
        (45, height // 2 + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        (255, 255, 255),
        5
    )
    return mask


# =========================
# 3. 覆盖静态图片
# =========================

def overlay_image(frame, mask_img, x, y, w, h):
    frame_h, frame_w = frame.shape[:2]

    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(frame_w, int(x + w))
    y2 = min(frame_h, int(y + h))

    if x2 <= x1 or y2 <= y1:
        return frame

    target_w = x2 - x1
    target_h = y2 - y1

    resized_mask = cv2.resize(mask_img, (target_w, target_h))

    # 支持透明 PNG
    if resized_mask.ndim == 3 and resized_mask.shape[2] == 4:
        bgr = resized_mask[:, :, :3]
        alpha = resized_mask[:, :, 3] / 255.0
        alpha = alpha[:, :, np.newaxis]

        roi = frame[y1:y2, x1:x2]
        blended = (alpha * bgr + (1 - alpha) * roi).astype(np.uint8)
        frame[y1:y2, x1:x2] = blended
    else:
        if resized_mask.ndim == 2:
            resized_mask = cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2BGR)

        frame[y1:y2, x1:x2] = resized_mask

    return frame


# =========================
# 4. 扩大遮挡框
# =========================

def expand_box(x, y, w, h, frame_w, frame_h):
    """
    HOG 检测出来的框有时只框到身体局部，
    所以这里把框适当放大，让遮挡效果更明显。
    """

    cx = x + w / 2
    cy = y + h / 2

    new_w = w * 1.8
    new_h = h * 1.8

    new_x = int(cx - new_w / 2)
    new_y = int(cy - new_h / 2)

    new_x = max(0, new_x)
    new_y = max(0, new_y)

    new_w = min(frame_w - new_x, int(new_w))
    new_h = min(frame_h - new_y, int(new_h))

    return new_x, new_y, new_w, new_h


# =========================
# 5. NMS 去重复框
# =========================

def nms_boxes(boxes, scores, score_threshold=0.4, nms_threshold=0.35):
    if len(boxes) == 0:
        return [], []

    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes,
        scores=scores,
        score_threshold=score_threshold,
        nms_threshold=nms_threshold
    )

    if len(indices) == 0:
        return [], []

    indices = np.array(indices).flatten()

    result_boxes = []
    result_scores = []

    for i in indices:
        result_boxes.append(boxes[i])
        result_scores.append(scores[i])

    return result_boxes, result_scores


# =========================
# 6. 过滤误检框
# =========================

def filter_person_boxes(boxes, scores, frame_w, frame_h):
    """
    过滤明显不像人的框：
    1. 太小的不要
    2. 太靠边的不要
    3. 长宽比例不像人的不要
    4. 置信度太低的不要
    """

    filtered_boxes = []
    filtered_scores = []

    for box, score in zip(boxes, scores):
        x, y, w, h = box

        area = w * h
        frame_area = frame_w * frame_h
        aspect_ratio = h / (w + 1e-6)

        # 1. 分数太低的不要
        if score < 0.45:
            continue

        # 2. 太小的框不要
        if area < frame_area * 0.025:
            continue

        # 3. 高度太小的不要
        if h < frame_h * 0.22:
            continue

        # 4. 宽度太小的不要
        if w < frame_w * 0.08:
            continue

        # 5. 人体一般是竖长形，比例太怪的不要
        if aspect_ratio < 1.1 or aspect_ratio > 4.5:
            continue

        # 6. 太靠左上角的框大概率是背景误检，过滤掉
        if x < frame_w * 0.08 and y < frame_h * 0.20:
            continue

        filtered_boxes.append(box)
        filtered_scores.append(score)

    return filtered_boxes, filtered_scores


# =========================
# 7. 选择最像目标人物的框
# =========================

def choose_best_target(boxes, scores, frame_w, frame_h):
    """
    如果有多个框，只选一个最像目标人物的框。
    规则：
    1. 分数越高越好
    2. 面积越大越好
    3. 越靠近画面中心越好
    """

    if len(boxes) == 0:
        return []

    frame_cx = frame_w / 2
    frame_cy = frame_h / 2

    best_idx = 0
    best_value = -1e9

    for i, (box, score) in enumerate(zip(boxes, scores)):
        x, y, w, h = box

        cx = x + w / 2
        cy = y + h / 2

        area_score = (w * h) / (frame_w * frame_h)

        distance = math.sqrt((cx - frame_cx) ** 2 + (cy - frame_cy) ** 2)
        max_distance = math.sqrt(frame_cx ** 2 + frame_cy ** 2)
        center_score = 1 - distance / max_distance

        value = score * 1.5 + area_score * 3.0 + center_score * 1.0

        if value > best_value:
            best_value = value
            best_idx = i

    return [boxes[best_idx]]


# =========================
# 8. 简单中心点跟踪器
# =========================

class CentroidTracker:
    def __init__(self, max_disappeared=8, max_distance=100):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, box):
        self.objects[self.next_object_id] = {
            "centroid": centroid,
            "box": box
        }
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]

    def update(self, boxes):
        if len(boxes) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1

                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            return self.objects

        input_centroids = []

        for (x, y, w, h) in boxes:
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            input_centroids.append((cx, cy))

        if len(self.objects) == 0:
            for centroid, box in zip(input_centroids, boxes):
                self.register(centroid, box)

            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[object_id]["centroid"] for object_id in object_ids]

        distance_matrix = np.zeros((len(object_centroids), len(input_centroids)), dtype=np.float32)

        for i, old_centroid in enumerate(object_centroids):
            for j, new_centroid in enumerate(input_centroids):
                distance_matrix[i, j] = math.dist(old_centroid, new_centroid)

        rows = distance_matrix.min(axis=1).argsort()
        cols = distance_matrix.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            if distance_matrix[row, col] > self.max_distance:
                continue

            object_id = object_ids[row]
            self.objects[object_id]["centroid"] = input_centroids[col]
            self.objects[object_id]["box"] = boxes[col]
            self.disappeared[object_id] = 0

            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(distance_matrix.shape[0])) - used_rows

        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1

            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        unused_cols = set(range(distance_matrix.shape[1])) - used_cols

        for col in unused_cols:
            self.register(input_centroids[col], boxes[col])

        return self.objects


# =========================
# 9. 主函数
# =========================

def main():
    mask_img = read_image_chinese_path(MASK_IMAGE_PATH, cv2.IMREAD_UNCHANGED)

    if mask_img is None:
        print("没有找到遮挡图片，使用默认 MASK 图片。")
        mask_img = create_default_mask()
    else:
        print("成功读取遮挡图片：", MASK_IMAGE_PATH)

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    tracker = CentroidTracker(max_disappeared=8, max_distance=100)

    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        print("摄像头打开失败。如果你有多个摄像头，可以把 CAMERA_ID 改成 1。")
        return

    print("摄像头已打开。按 q 退出。")
    print("提示：HOG 更适合检测完整站立的人，建议人离摄像头远一点。")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("读取摄像头失败。")
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))

        frame_h, frame_w = frame.shape[:2]

        # HOG 行人检测
        boxes, weights = hog.detectMultiScale(
            frame,
            winStride=(4, 4),
            padding=(8, 8),
            scale=1.03
        )

        boxes = boxes.tolist() if len(boxes) > 0 else []

        if len(weights) > 0:
            scores = [float(w[0]) if isinstance(w, (list, np.ndarray)) else float(w) for w in weights]
        else:
            scores = []

        # 显示原始检测框，方便调试
        if SHOW_RAW_BOXES:
            for (x, y, w, h) in boxes:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 1)

        # 先 NMS 去重
        boxes, scores = nms_boxes(
            boxes,
            scores,
            score_threshold=0.35,
            nms_threshold=0.35
        )

        # 再过滤误检框
        boxes, scores = filter_person_boxes(
            boxes,
            scores,
            frame_w,
            frame_h
        )

        # 只选最像目标人的一个框
        if not COVER_ALL_PEOPLE:
            boxes = choose_best_target(boxes, scores, frame_w, frame_h)

        # 跟踪
        objects = tracker.update(boxes)

        # 绘制和遮挡
        for object_id, info in objects.items():
            x, y, w, h = info["box"]
            cx, cy = info["centroid"]

            # 原始检测框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"ID {object_id}",
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            # 扩大遮挡区域
            ex, ey, ew, eh = expand_box(x, y, w, h, frame_w, frame_h)

            # 用静态图片遮挡
            frame = overlay_image(frame, mask_img, ex, ey, ew, eh)

        cv2.putText(
            frame,
            f"Pedestrians: {len(objects)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 0, 0),
            2
        )

        cv2.imshow("HOG Pedestrian Detection and Static Mask V2", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == QUIT_KEY:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()