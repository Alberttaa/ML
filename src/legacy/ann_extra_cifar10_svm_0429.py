import os
import time
import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# ======================================================
# 附加题简略思路：
# 1. 读取 CIFAR-10 的 train 和 test 文件夹，10 个类别分别编号 0-9；
# 2. 构建三层以上 CNN 神经网络；
# 3. 实验一：随机初始化网络，不训练，只提取最后一层特征，用 SVM 分类；
# 4. 实验二：正常训练神经网络，直接用神经网络输出分类，计算测试准确率；
# 5. 实验三：训练好的神经网络提取最后一层特征，再用 SVM 分类；
# 6. 输出三种方法的测试准确率，并绘制 ANN 训练 loss 曲线。
# ======================================================


# =========================
# 1. 路径配置
# =========================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "cifar10")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

ANN_MODEL_PATH = os.path.join(MODEL_DIR, "ann_cifar10.pth")
SVM_RANDOM_PATH = os.path.join(MODEL_DIR, "svm_random_ann_features.pkl")
SVM_TRAINED_PATH = os.path.join(MODEL_DIR, "svm_trained_ann_features.pkl")


# =========================
# 2. 参数配置
# =========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
EPOCHS = 15
LR = 0.001

# 如果电脑运行慢，可以限制数据量
# None 表示使用全部数据
TRAIN_LIMIT = 10000
TEST_LIMIT = 2000

# SVM 用 LinearSVC，速度比普通 SVC 快很多
SVM_MAX_ITER = 5000

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CLASS_TO_LABEL = {name: i for i, name in enumerate(CLASS_NAMES)}


# =========================
# 3. 创建文件夹
# =========================

def make_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)


# =========================
# 4. 支持中文路径读取图片
# =========================

def read_image(path):
    try:
        img_array = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


# =========================
# 5. 自定义 CIFAR-10 数据集
# =========================

class Cifar10FolderDataset(Dataset):
    def __init__(self, root_dir, limit=None):
        self.root_dir = root_dir
        self.samples = []

        image_exts = [".jpg", ".jpeg", ".png", ".bmp"]

        for class_name in CLASS_NAMES:
            class_dir = os.path.join(root_dir, class_name)

            if not os.path.exists(class_dir):
                print("警告：找不到类别文件夹：", class_dir)
                continue

            label = CLASS_TO_LABEL[class_name]

            for filename in os.listdir(class_dir):
                ext = os.path.splitext(filename)[1].lower()

                if ext not in image_exts:
                    continue

                img_path = os.path.join(class_dir, filename)
                self.samples.append((img_path, label))

                if limit is not None and len(self.samples) >= limit:
                    break

            if limit is not None and len(self.samples) >= limit:
                break

        print(f"{root_dir} 读取图片数量：{len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        img = read_image(img_path)

        if img is None:
            # 防止坏图导致程序中断
            img = np.zeros((32, 32, 3), dtype=np.uint8)

        img = cv2.resize(img, (32, 32))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 转成 float，并归一化到 [0,1]
        img = img.astype(np.float32) / 255.0

        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))

        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# =========================
# 6. 构建三层以上 ANN/CNN
# =========================

class CifarCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # 卷积层部分：多层特征提取
        self.conv = nn.Sequential(
            # 第1层
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 第2层
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 第3层
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 全连接层部分
        self.fc_feature = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            # 最后输出的特征向量，后面给 SVM 使用
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # 分类输出层
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x, return_feature=False):
        x = self.conv(x)
        x = x.view(x.size(0), -1)

        feature = self.fc_feature(x)
        logits = self.classifier(feature)

        if return_feature:
            return logits, feature

        return logits


# =========================
# 7. 训练 ANN
# =========================

def train_ann(model, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses = []
    test_accs = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_num = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_num += images.size(0)

        avg_loss = total_loss / total_num
        train_losses.append(avg_loss)

        test_acc = evaluate_ann(model, test_loader)
        test_accs.append(test_acc)

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] "
            f"loss={avg_loss:.4f} "
            f"test_acc={test_acc:.4f}"
        )

    torch.save(model.state_dict(), ANN_MODEL_PATH)
    print("ANN 模型已保存：", ANN_MODEL_PATH)

    return train_losses, test_accs


# =========================
# 8. 测试 ANN 准确率
# =========================

@torch.no_grad()
def evaluate_ann(model, data_loader):
    model.eval()

    all_preds = []
    all_labels = []

    for images, labels in data_loader:
        images = images.to(DEVICE)

        logits = model(images)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)

    return acc


# =========================
# 9. 提取神经网络最后特征向量
# =========================

@torch.no_grad()
def extract_ann_features(model, data_loader):
    model.eval()

    features = []
    labels_all = []

    for images, labels in data_loader:
        images = images.to(DEVICE)

        _, feature = model(images, return_feature=True)

        features.append(feature.cpu().numpy())
        labels_all.append(labels.numpy())

    features = np.vstack(features)
    labels_all = np.concatenate(labels_all)

    return features, labels_all


# =========================
# 10. 用 SVM 训练和测试
# =========================

def train_test_svm(x_train, y_train, x_test, y_test, save_path, name):
    """
    SVM 训练分类：
    先标准化，再用 LinearSVC。
    """
    svm_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(max_iter=SVM_MAX_ITER))
    ])

    start_time = time.time()

    svm_model.fit(x_train, y_train)
    y_pred = svm_model.predict(x_test)

    end_time = time.time()

    acc = accuracy_score(y_test, y_pred)

    joblib.dump(svm_model, save_path)

    print(f"{name} 测试准确率：{acc:.4f}")
    print(f"{name} SVM 用时：{end_time - start_time:.2f} 秒")
    print(f"{name} 模型已保存：{save_path}")

    return acc


# =========================
# 11. 绘图
# =========================

def plot_loss_curve(losses):
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ANN Training Loss Curve")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, "ann_cifar10_loss_curve.png")
    plt.savefig(save_path, dpi=300)
    print("loss 曲线已保存：", save_path)
    plt.show()


def plot_accuracy_bar(acc_random_svm, acc_ann, acc_trained_svm):
    names = [
        "Random ANN Feature + SVM",
        "Trained ANN Direct",
        "Trained ANN Feature + SVM"
    ]

    accs = [acc_random_svm, acc_ann, acc_trained_svm]

    plt.figure(figsize=(10, 5))
    plt.bar(names, accs)
    plt.ylim(0, 1)
    plt.ylabel("Test Accuracy")
    plt.title("CIFAR-10 Accuracy Comparison")

    for i, acc in enumerate(accs):
        plt.text(i, acc + 0.02, f"{acc:.4f}", ha="center")

    plt.xticks(rotation=15)
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, "ann_svm_accuracy_comparison.png")
    plt.savefig(save_path, dpi=300)
    print("准确率对比图已保存：", save_path)
    plt.show()


def save_result_text(acc_random_svm, acc_ann, acc_trained_svm):
    save_path = os.path.join(RESULTS_DIR, "ann_extra_cifar10_svm_result.txt")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("ANN 附加题 CIFAR-10 实验结果\n")
        f.write("=" * 50 + "\n\n")

        f.write("实验1：随机神经网络参数，不训练，提取最后特征向量，用 SVM 分类\n")
        f.write(f"测试准确率：{acc_random_svm:.4f}\n\n")

        f.write("实验2：神经网络训练收敛后，直接在测试集分类\n")
        f.write(f"测试准确率：{acc_ann:.4f}\n\n")

        f.write("实验3：神经网络训练收敛后，提取最后特征向量，用 SVM 分类\n")
        f.write(f"测试准确率：{acc_trained_svm:.4f}\n\n")

    print("实验结果文本已保存：", save_path)


# =========================
# 12. 主函数
# =========================

def main():
    make_dirs()

    print("当前设备：", DEVICE)
    print("训练集路径：", TRAIN_DIR)
    print("测试集路径：", TEST_DIR)

    train_dataset = Cifar10FolderDataset(TRAIN_DIR, limit=TRAIN_LIMIT)
    test_dataset = Cifar10FolderDataset(TEST_DIR, limit=TEST_LIMIT)

    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print("没有读到训练集或测试集，请检查 data/cifar10/train 和 data/cifar10/test。")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    train_feature_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_feature_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # ==================================================
    # 实验1：随机神经网络参数，不训练，特征 + SVM
    # ==================================================

    print("\n" + "=" * 70)
    print("实验1：随机 ANN 参数，不训练，提取特征后用 SVM 分类")
    print("=" * 70)

    random_model = CifarCNN(num_classes=10).to(DEVICE)

    x_train_random, y_train_random = extract_ann_features(random_model, train_feature_loader)
    x_test_random, y_test_random = extract_ann_features(random_model, test_feature_loader)

    acc_random_svm = train_test_svm(
        x_train_random,
        y_train_random,
        x_test_random,
        y_test_random,
        save_path=SVM_RANDOM_PATH,
        name="Random ANN Feature + SVM"
    )

    # ==================================================
    # 实验2：训练 ANN，直接分类
    # ==================================================

    print("\n" + "=" * 70)
    print("实验2：训练 ANN，直接在测试集分类")
    print("=" * 70)

    trained_model = CifarCNN(num_classes=10).to(DEVICE)

    losses, test_accs = train_ann(
        model=trained_model,
        train_loader=train_loader,
        test_loader=test_loader
    )

    acc_ann = test_accs[-1]

    plot_loss_curve(losses)

    print(f"Trained ANN Direct 测试准确率：{acc_ann:.4f}")

    # ==================================================
    # 实验3：训练后的 ANN 特征 + SVM
    # ==================================================

    print("\n" + "=" * 70)
    print("实验3：训练后的 ANN 提取特征，再用 SVM 分类")
    print("=" * 70)

    x_train_trained, y_train_trained = extract_ann_features(trained_model, train_feature_loader)
    x_test_trained, y_test_trained = extract_ann_features(trained_model, test_feature_loader)

    acc_trained_svm = train_test_svm(
        x_train_trained,
        y_train_trained,
        x_test_trained,
        y_test_trained,
        save_path=SVM_TRAINED_PATH,
        name="Trained ANN Feature + SVM"
    )

    # ==================================================
    # 保存和展示结果
    # ==================================================

    print("\n" + "=" * 70)
    print("最终结果汇总")
    print("=" * 70)
    print(f"随机 ANN 特征 + SVM 测试准确率：{acc_random_svm:.4f}")
    print(f"训练 ANN 直接分类测试准确率：{acc_ann:.4f}")
    print(f"训练 ANN 特征 + SVM 测试准确率：{acc_trained_svm:.4f}")

    save_result_text(acc_random_svm, acc_ann, acc_trained_svm)
    plot_accuracy_bar(acc_random_svm, acc_ann, acc_trained_svm)


if __name__ == "__main__":
    main()