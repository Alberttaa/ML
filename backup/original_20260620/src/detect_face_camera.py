import os
import cv2
import torch
import numpy as np

from PIL import Image
from torchvision import transforms, models
import torch.nn as nn


# =========================
# 路径
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "face_vgg16.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
model = models.vgg16(pretrained=False)

model.classifier[6] = nn.Linear(4096, 2)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model = model.to(device)

model.eval()

print("模型加载成功")

# =========================
# 摄像头
# =========================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# patch大小
PATCH_SIZE = 128


# 滑动步长
STRIDE = 72

while True:

    ret, frame = cap.read()

    if not ret:
        break

    img_h, img_w, _ = frame.shape

    # 遍历patch
    for y in range(0, img_h - PATCH_SIZE, STRIDE):

        for x in range(0, img_w - PATCH_SIZE, STRIDE):

            patch = frame[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            # BGR -> RGB
            patch_rgb = cv2.cvtColor(
                patch,
                cv2.COLOR_BGR2RGB
            )

            pil_img = Image.fromarray(patch_rgb)

            input_tensor = transform(pil_img)

            input_tensor = input_tensor.unsqueeze(0).to(device)

            with torch.no_grad():

                output = model(input_tensor)

                prob = torch.softmax(output, dim=1)

                pred = torch.argmax(prob, dim=1).item()

                confidence = prob[0][pred].item()

            # 1 = positive = face
            if pred == 1 and confidence > 0.93:

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + PATCH_SIZE, y + PATCH_SIZE),
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Face {confidence:.2f}",
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1
                )

    cv2.imshow("Face Detection", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()

cv2.destroyAllWindows()