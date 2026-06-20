import os
import cv2
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms, models
from torchvision.models import VGG16_Weights


# =====================
# 路径设置
# =====================
# 如果你的代码在 src 文件夹里，用这一句：
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 如果你的代码直接放在项目根目录，就改成下面这一句：
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "models",
    "standard_face_feature.pt"
)


# =====================
# 设备
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前设备:", device)


# =====================
# 加载 VGG16
# 不训练，直接使用原始 1000 维输出
# =====================
model = models.vgg16(weights=VGG16_Weights.DEFAULT)
model = model.to(device)
model.eval()


# =====================
# 加载标准人脸特征
# =====================
if not os.path.exists(FEATURE_PATH):
    raise FileNotFoundError(
        f"没有找到标准人脸特征文件: {FEATURE_PATH}\n"
        f"请先生成 standard_face_feature.pt"
    )

face_feature = torch.load(FEATURE_PATH, map_location=device)

# 统一成 [1, 1000]
if face_feature.dim() == 1:
    face_feature = face_feature.unsqueeze(0)

face_feature = face_feature.to(device)
face_feature = F.normalize(face_feature, dim=1)


# =====================
# 图像预处理
# 必须和生成标准人脸特征时保持一致
# =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =====================
# 生成 patch 坐标
# 这样可以保证最右边、最下面也被覆盖到
# =====================
def get_positions(length, patch_size, stride):
    positions = list(range(0, length - patch_size + 1, stride))

    last_pos = length - patch_size
    if last_pos > 0 and last_pos not in positions:
        positions.append(last_pos)

    return positions


# =====================
# 摄像头
# =====================
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("摄像头打开失败，请检查摄像头权限或设备编号。")

PATCH_SIZE = 224
STRIDE = 112

print("按 ESC 退出程序。")

while True:
    ret, frame = cap.read()

    if not ret:
        print("读取摄像头画面失败。")
        break

    h, w, _ = frame.shape

    # 如果画面太小，无法切出 224x224 的 patch
    if h < PATCH_SIZE or w < PATCH_SIZE:
        cv2.imshow("Cosine Face Detection", frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    xs = get_positions(w, PATCH_SIZE, STRIDE)
    ys = get_positions(h, PATCH_SIZE, STRIDE)

    patch_tensors = []
    boxes = []

    # =====================
    # 划分所有 patch
    # =====================
    for y in ys:
        for x in xs:
            patch = frame[y:y + PATCH_SIZE, x:x + PATCH_SIZE]

            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(patch_rgb)

            tensor = transform(img)
            patch_tensors.append(tensor)

            boxes.append((
                x,
                y,
                x + PATCH_SIZE,
                y + PATCH_SIZE
            ))

    if len(patch_tensors) > 0:
        batch = torch.stack(patch_tensors, dim=0).to(device)

        with torch.no_grad():
            # 输出 shape: [N, 1000]
            patch_features = model(batch)

            # 归一化后再算余弦相似度
            patch_features = F.normalize(patch_features, dim=1)

            # 每个 patch 和标准脸特征算余弦相似度
            scores = F.cosine_similarity(
                patch_features,
                face_feature.expand_as(patch_features),
                dim=1
            )

            # 余弦距离最近 <=> 余弦相似度最大
            best_index = torch.argmax(scores).item()
            best_score = scores[best_index].item()
            best_box = boxes[best_index]

        # =====================
        # 只画一个最佳 patch 框
        # =====================
        x1, y1, x2, y2 = best_box

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3
        )

        cv2.putText(
            frame,
            f"Best Face Patch cos={best_score:.3f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    cv2.imshow("Cosine Face Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()