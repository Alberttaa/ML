import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt


# =========================
# 路径
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "face_dataset")

MODEL_PATH = os.path.join(BASE_DIR, "models", "face_vgg16.pth")

os.makedirs(
    os.path.join(BASE_DIR, "models"),
    exist_ok=True
)
BATCH_SIZE = 4
EPOCHS = 10
LR = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("当前设备：", device)

# =========================
# 数据增强
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(15),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# =========================
# 数据集
# =========================
dataset = datasets.ImageFolder(
    root=DATA_DIR,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print("类别：", dataset.classes)
print("总图片数：", len(dataset))

# =========================
# VGG16
from torchvision.models import VGG16_Weights

model = models.vgg16(
    weights=VGG16_Weights.DEFAULT
)
# 冻结参数
for param in model.parameters():
    param.requires_grad = False

# 修改最后一层
model.classifier[6] = nn.Linear(4096, 2)

model = model.to(device)

# =========================
# Loss 和优化器
# =========================
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.classifier[6].parameters(),
    lr=LR
)

# =========================
# 训练
# =========================
loss_history = []
acc_history = []

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        _, pred = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (pred == labels).sum().item()

    avg_loss = total_loss / len(loader)

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
    MODEL_PATH,
    _use_new_zipfile_serialization=False
)
print("模型已保存：", MODEL_PATH)

# =========================
# 画图
# =========================
plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.title("VGG16 Face Classification Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(acc_history)
plt.title("VGG16 Face Classification Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()