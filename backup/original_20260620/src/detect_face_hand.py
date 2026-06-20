import os
import cv2
import torch
import torch.nn as nn

from PIL import Image

from torchvision import transforms
from torchvision import models

# =========================
# 路径
# =========================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "vgg_face_hand.pth"
)

# =========================
# 设备
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("当前设备：", device)

# =========================
# 图像预处理
# =========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# =========================
# 加载模型
# =========================

model = models.vgg16(weights=None)

model.classifier[6] = nn.Linear(
    4096,
    3
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.to(device)

model.eval()

print("模型加载成功")

# =========================
# 类别名称
# =========================

# 请根据训练时输出的 dataset.classes 修改
# 如果训练输出：
# ['background', 'face', 'hand']
# 就保持下面不变

CLASS_NAMES = [
    "Background",
    "Face",
    "Hand"
]

# =========================
# 摄像头
# =========================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# =========================
# 滑窗参数
# =========================

PATCH_SIZE = 224
STRIDE = 112

# =========================
# 主循环
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    h, w, _ = frame.shape

    face_boxes = []
    hand_boxes = []

    # ---------------------
    # 滑窗扫描
    # ---------------------

    for y in range(
        0,
        h - PATCH_SIZE,
        STRIDE
    ):

        for x in range(
            0,
            w - PATCH_SIZE,
            STRIDE
        ):

            patch = frame[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            patch_rgb = cv2.cvtColor(
                patch,
                cv2.COLOR_BGR2RGB
            )

            img = Image.fromarray(
                patch_rgb
            )

            tensor = transform(
                img
            )

            tensor = tensor.unsqueeze(0)

            tensor = tensor.to(device)

            with torch.no_grad():

                output = model(
                    tensor
                )

                prob = torch.softmax(
                    output,
                    dim=1
                )

                pred = torch.argmax(
                    prob,
                    dim=1
                ).item()

                confidence = (
                    prob[0][pred]
                    .item()
                )

            # Background
            if pred == 0:
                continue

            # Face
            if pred == 1 and confidence > 0.90:

                face_boxes.append(
                    (
                        x,
                        y,
                        x + PATCH_SIZE,
                        y + PATCH_SIZE
                    )
                )

            # Hand
            elif pred == 2 and confidence > 0.90:

                hand_boxes.append(
                    (
                        x,
                        y,
                        x + PATCH_SIZE,
                        y + PATCH_SIZE
                    )
                )

    # =====================
    # Face框合并
    # =====================

    if len(face_boxes) > 0:

        x1 = min(box[0] for box in face_boxes)
        y1 = min(box[1] for box in face_boxes)

        x2 = max(box[2] for box in face_boxes)
        y2 = max(box[3] for box in face_boxes)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255,0),
            3
        )

        cv2.putText(
            frame,
            "Face",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    # =====================
    # Hand框合并
    # =====================

    if len(hand_boxes) > 0:

        x1 = min(box[0] for box in hand_boxes)
        y1 = min(box[1] for box in hand_boxes)

        x2 = max(box[2] for box in hand_boxes)
        y2 = max(box[3] for box in hand_boxes)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            3
        )

        cv2.putText(
            frame,
            "Hand",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2
        )

    # =====================
    # 显示结果
    # =====================

    cv2.imshow(
        "Face & Hand Detection",
        frame
    )

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()