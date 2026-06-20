import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
from torchvision import transforms
from torchvision import models
from torchvision.models import VGG16_Weights

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# =========================
# 路径
# =========================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "face_hand"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "vgg_face_hand.pth"
)

os.makedirs(
    os.path.join(BASE_DIR, "models"),
    exist_ok=True
)

# =========================
# 设备
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("当前设备：", device)

# =========================
# 数据增强
# =========================

transform = transforms.Compose([
    transforms.Resize((224,224)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(15),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# =========================
# 数据集
# =========================

dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

print("类别：", dataset.classes)
print("总图片数：", len(dataset))

# =========================
# VGG16
# =========================

model = models.vgg16(
    weights=VGG16_Weights.DEFAULT
)

# 冻结卷积层

for param in model.features.parameters():
    param.requires_grad = False

# 修改输出层

model.classifier[6] = nn.Linear(
    4096,
    3
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.classifier.parameters(),
    lr=0.001
)

# =========================
# 训练
# =========================

loss_history = []
acc_history = []

EPOCHS = 10

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, pred = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            pred == labels
        ).sum().item()

    avg_loss = running_loss / len(loader)

    acc = correct / total

    loss_history.append(avg_loss)
    acc_history.append(acc)

    print(
        f"Epoch {epoch+1}/{EPOCHS} "
        f"Loss={avg_loss:.4f} "
        f"Accuracy={acc:.4f}"
    )

# =========================
# 保存模型
# =========================

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print("\n模型保存成功：")
print(MODEL_PATH)

# =========================
# Loss曲线
# =========================

plt.figure(figsize=(8,5))
plt.plot(loss_history)
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.show()

# =========================
# Accuracy曲线
# =========================

plt.figure(figsize=(8,5))
plt.plot(acc_history)
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid()
plt.show()