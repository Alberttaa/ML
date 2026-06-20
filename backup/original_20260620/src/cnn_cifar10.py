import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import font_manager

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

font_path = "C:/Windows/Fonts/msyh.ttc"
my_font = font_manager.FontProperties(fname=font_path)
plt.rcParams["font.family"] = my_font.get_name()
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

EPOCHS = 5
BATCH_SIZE = 64
LR = 0.001


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("当前设备：", device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=os.path.join(BASE_DIR, "data"),
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=os.path.join(BASE_DIR, "data"),
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    loss_history = []
    acc_history = []

    print("开始训练 CNN CIFAR10 模型...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        train_acc = correct / total

        loss_history.append(avg_loss)
        acc_history.append(train_acc)

        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss={avg_loss:.4f}, Train Accuracy={train_acc:.4f}")

    print("\n开始测试 CNN 模型...")
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = correct / total

    print("\n========== CNN CIFAR10 测试结果 ==========")
    print(f"测试准确率：{test_acc:.4f}")

    model_path = os.path.join(MODELS_DIR, "cnn_cifar10.pth")
    torch.save(model.state_dict(), model_path)
    print("模型保存到：", model_path)

    plt.figure(figsize=(8, 5))
    plt.plot(loss_history)
    plt.xlabel("训练轮数 Epoch")
    plt.ylabel("Loss")
    plt.title("CNN CIFAR10 Loss曲线")
    plt.grid(True)
    loss_path = os.path.join(RESULTS_DIR, "cnn_cifar10_loss.png")
    plt.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(acc_history)
    plt.xlabel("训练轮数 Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN CIFAR10训练Accuracy曲线")
    plt.grid(True)
    acc_path = os.path.join(RESULTS_DIR, "cnn_cifar10_accuracy.png")
    plt.savefig(acc_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("Loss曲线保存到：", loss_path)
    print("Accuracy曲线保存到：", acc_path)


if __name__ == "__main__":
    run()