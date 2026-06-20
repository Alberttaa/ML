import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import font_manager


# =========================
# 0. 基础设置
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "ann", "data0515.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# 中文字体
font_path = "C:/Windows/Fonts/msyh.ttc"
if os.path.exists(font_path):
    my_font = font_manager.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = my_font.get_name()

plt.rcParams["axes.unicode_minus"] = False

# 超参数
SEED = 42
EPOCHS = 100
BATCH_SIZE = 64
LR = 0.01

INPUT_DIM = 10
NUM_CLASSES = 10


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =========================
# 1. 读取数据
# =========================
def load_data():
    print("正在读取数据：", DATA_PATH)

    df = pd.read_csv(DATA_PATH)
    print("数据形状：", df.shape)
    print("数据列名：", df.columns.tolist())
    print("标签类别：", sorted(df["label"].unique().tolist()))

    feature_cols = [f"feature_{i}" for i in range(10)]

    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int64)

    # 标准化，用训练集整体统计量即可，因为本题不划分训练测试
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1
    X = (X - mean) / std

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    return loader, X_tensor, y_tensor


# =========================
# 2. 网络一：普通分类网络，可选BN
# =========================
class ClassifierNet(nn.Module):
    def __init__(self, use_bn=False):
        super().__init__()

        if use_bn:
            self.net = nn.Sequential(
                nn.Linear(INPUT_DIM, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),

                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),

                nn.Linear(32, NUM_CLASSES)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(INPUT_DIM, 64),
                nn.ReLU(),

                nn.Linear(64, 32),
                nn.ReLU(),

                nn.Linear(32, NUM_CLASSES)
            )

    def forward(self, x):
        return self.net(x)


# =========================
# 3. 网络二：单神经元输出，用MSE和0~9标签回归
# =========================
class SingleOutputNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# 4. 训练：CrossEntropy分类方案
# =========================
def train_ce_model(model, loader, x_all, y_all, epochs=EPOCHS, lr=LR):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    acc_history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for xb, yb in loader:
            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(loader.dataset)

        model.eval()
        with torch.no_grad():
            logits_all = model(x_all)
            pred = torch.argmax(logits_all, dim=1)
            acc = (pred == y_all).float().mean().item()

        loss_history.append(avg_loss)
        acc_history.append(acc)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:03d}, Loss={avg_loss:.4f}, Accuracy={acc:.4f}")

    return loss_history, acc_history


# =========================
# 5. 训练：单神经元 + MSE方案
# =========================
def train_mse_single_output(model, loader, x_all, y_all, epochs=EPOCHS, lr=LR):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    acc_history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for xb, yb in loader:
            output = model(xb).squeeze(1)

            # 标签从 0~9 转成 float，直接做MSE
            y_float = yb.float()
            loss = criterion(output, y_float)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(loader.dataset)

        model.eval()
        with torch.no_grad():
            output_all = model(x_all).squeeze(1)

            # 单神经元输出浮点数，四舍五入到0~9作为类别
            pred = torch.round(output_all).long()
            pred = torch.clamp(pred, 0, 9)

            acc = (pred == y_all).float().mean().item()

        loss_history.append(avg_loss)
        acc_history.append(acc)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:03d}, Loss={avg_loss:.4f}, Accuracy={acc:.4f}")

    return loss_history, acc_history


# =========================
# 6. 实验1：BN vs 不加BN
# =========================
def experiment_bn_compare(loader, x_all, y_all):
    print("\n========== 实验1：BatchNorm 对比 ==========")

    set_seed(SEED)
    model_no_bn = ClassifierNet(use_bn=False)

    print("\n开始训练：不加BN")
    loss_no_bn, acc_no_bn = train_ce_model(model_no_bn, loader, x_all, y_all)

    set_seed(SEED)
    model_bn = ClassifierNet(use_bn=True)

    print("\n开始训练：加入BN")
    loss_bn, acc_bn = train_ce_model(model_bn, loader, x_all, y_all)

    print("\n========== BN对比最终结果 ==========")
    print(f"不加BN最终准确率：{acc_no_bn[-1]:.4f}")
    print(f"加入BN最终准确率：{acc_bn[-1]:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(loss_no_bn, label=f"不加BN Loss，最终Acc={acc_no_bn[-1]:.4f}")
    plt.plot(loss_bn, label=f"加入BN Loss，最终Acc={acc_bn[-1]:.4f}")
    plt.xlabel("训练轮数 Epoch")
    plt.ylabel("Loss")
    plt.title("有BN与无BN的Loss曲线对比")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(RESULTS_DIR, "ann3_bn_loss_compare.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    torch.save(model_no_bn.state_dict(), os.path.join(MODELS_DIR, "ann3_no_bn.pth"))
    torch.save(model_bn.state_dict(), os.path.join(MODELS_DIR, "ann3_with_bn.pth"))

    print("BN对比图保存到：", save_path)


# =========================
# 7. 实验2：CE vs 单神经元MSE
# =========================
def experiment_ce_vs_mse(loader, x_all, y_all):
    print("\n========== 实验2：Softmax+CE 与 单神经元+MSE 对比 ==========")

    set_seed(SEED)
    ce_model = ClassifierNet(use_bn=False)

    print("\n开始训练：Softmax + CrossEntropy")
    ce_loss, ce_acc = train_ce_model(ce_model, loader, x_all, y_all)

    set_seed(SEED)
    mse_model = SingleOutputNet()

    print("\n开始训练：单神经元输出 + MSE")
    mse_loss, mse_acc = train_mse_single_output(mse_model, loader, x_all, y_all)

    print("\n========== CE vs MSE最终结果 ==========")
    print(f"Softmax+CE最终准确率：{ce_acc[-1]:.4f}")
    print(f"单神经元+MSE最终准确率：{mse_acc[-1]:.4f}")

    # Loss 曲线对比
    plt.figure(figsize=(8, 5))
    plt.plot(ce_loss, label=f"Softmax+CE Loss，最终Acc={ce_acc[-1]:.4f}")
    plt.plot(mse_loss, label=f"单神经元+MSE Loss，最终Acc={mse_acc[-1]:.4f}")
    plt.xlabel("训练轮数 Epoch")
    plt.ylabel("Loss")
    plt.title("Softmax+CE 与 单神经元+MSE 的Loss曲线对比")
    plt.legend()
    plt.grid(True)

    loss_path = os.path.join(RESULTS_DIR, "ann3_ce_vs_mse_loss.png")
    plt.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.show()

    # Accuracy 曲线对比
    plt.figure(figsize=(8, 5))
    plt.plot(ce_acc, label="Softmax+CE Accuracy")
    plt.plot(mse_acc, label="单神经元+MSE Accuracy")
    plt.xlabel("训练轮数 Epoch")
    plt.ylabel("Accuracy")
    plt.title("Softmax+CE 与 单神经元+MSE 的Accuracy曲线对比")
    plt.legend()
    plt.grid(True)

    acc_path = os.path.join(RESULTS_DIR, "ann3_ce_vs_mse_accuracy.png")
    plt.savefig(acc_path, dpi=300, bbox_inches="tight")
    plt.show()

    torch.save(ce_model.state_dict(), os.path.join(MODELS_DIR, "ann3_ce.pth"))
    torch.save(mse_model.state_dict(), os.path.join(MODELS_DIR, "ann3_mse_single_output.pth"))

    print("Loss对比图保存到：", loss_path)
    print("Accuracy对比图保存到：", acc_path)


# =========================
# 8. 主函数
# =========================
def run():
    set_seed(SEED)

    loader, x_all, y_all = load_data()

    experiment_bn_compare(loader, x_all, y_all)
    experiment_ce_vs_mse(loader, x_all, y_all)

    print("\n全部实验完成。")


if __name__ == "__main__":
    run()