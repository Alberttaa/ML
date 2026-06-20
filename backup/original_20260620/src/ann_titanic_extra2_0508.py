import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader


# ======================================================
# 附加题简略思路：
# 1. 读取 Titanic 数据，自动识别标签列 Survived / 2urvived；
# 2. 删除无效列、编号列，对类别特征 one-hot，对数值特征归一化；
# 3. 构建至少 5 层隐藏层的 ANN，每层都是 Linear + 激活函数；
# 4. 固定学习率，分别使用 sigmoid、tanh、relu、leaky-relu；
# 5. 记录每种激活函数的 loss 曲线、训练准确率和测试准确率；
# 6. 对 ReLU 统计每一层失活神经元：如果某神经元在全部训练样本上输出都为 0，则认为失活；
# 7. 将 ReLU 学习率扩大 10 倍、100 倍，观察失活神经元数量变化。
# ======================================================


# =========================
# 1. 路径和参数配置
# =========================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# 你可以把数据放在 My project/data/titanic_train_knn.csv
DATA_PATH = r"E:\Ling\homework\aaa\My project\data\titanic\titanic_test_knn.csv"

# 如果你的路径不是上面这个，就直接改成绝对路径，例如：
# DATA_PATH = r"E:\Ling\homework\aaa\data_titanic\titanic_train_knn.csv"

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

SEED = 42
BATCH_SIZE = 32
EPOCHS = 120

# 基础学习率
BASE_LR = 0.001

# 至少 5 层隐藏层
HIDDEN_DIMS = [64, 64, 32, 32, 16]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 2. 固定随机种子
# =========================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================
# 3. 数据读取和清洗
# =========================

def read_csv_safely(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到数据文件：{path}\n"
            f"请检查 DATA_PATH 是否正确。"
        )

    for enc in ["utf-8", "gbk", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue

    return pd.read_csv(path)


def preprocess_titanic(df):
    """
    自动处理 Titanic 数据：
    - 标签列支持 Survived / survived / 2urvived；
    - 删除编号列、无效 zero 列；
    - 类别特征 one-hot；
    - 数值特征缺失值用中位数填充。
    """

    df = df.copy()

    possible_label_cols = ["Survived", "survived", "2urvived", "target", "label"]

    label_col = None
    for col in possible_label_cols:
        if col in df.columns:
            label_col = col
            break

    if label_col is None:
        raise ValueError(
            "没有找到标签列。请检查数据里是否有 Survived 或 2urvived。\n"
            f"当前列名：{list(df.columns)}"
        )

    print("检测到标签列：", label_col)

    df = df.dropna(subset=[label_col])
    y = df[label_col].astype(np.float32).values.reshape(-1, 1)

    drop_cols = [label_col]

    for col in ["PassengerId", "Passengerid", "passengerid", "Name", "Ticket", "Cabin"]:
        if col in df.columns:
            drop_cols.append(col)

    X = df.drop(columns=drop_cols)

    # 删除只有一个取值的无效列，比如 zero、zero.1
    nunique = X.nunique(dropna=False)
    useless_cols = nunique[nunique <= 1].index.tolist()

    if len(useless_cols) > 0:
        print("删除无效列：", useless_cols)
        X = X.drop(columns=useless_cols)

    # 区分数值列和类别列
    numeric_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    # 数值列缺失值处理
    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].fillna(X[col].median())

    # 类别列 one-hot
    for col in categorical_cols:
        X[col] = X[col].fillna("Unknown").astype(str)

    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols)

    X = X.astype(np.float32)

    print("最终特征维度：", X.shape)
    print("最终特征列：")
    print(list(X.columns))

    return X.values, y


# =========================
# 4. ANN 模型
# =========================

class TitanicANN(nn.Module):
    def __init__(self, input_dim, activation_name="relu"):
        super().__init__()

        self.hidden_linears = nn.ModuleList()
        self.hidden_activations = nn.ModuleList()

        last_dim = input_dim

        for hidden_dim in HIDDEN_DIMS:
            self.hidden_linears.append(nn.Linear(last_dim, hidden_dim))
            self.hidden_activations.append(self.get_activation(activation_name))
            last_dim = hidden_dim

        self.output_layer = nn.Linear(last_dim, 1)

    def get_activation(self, activation_name):
        activation_name = activation_name.lower()

        if activation_name == "sigmoid":
            return nn.Sigmoid()
        elif activation_name == "tanh":
            return nn.Tanh()
        elif activation_name == "relu":
            return nn.ReLU()
        elif activation_name in ["leaky_relu", "leaky-relu", "leakyrelu"]:
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            raise ValueError(f"未知激活函数：{activation_name}")

    def forward(self, x, return_activations=False):
        activations = []

        for linear_layer, activation_layer in zip(self.hidden_linears, self.hidden_activations):
            x = linear_layer(x)
            x = activation_layer(x)

            if return_activations:
                activations.append(x)

        logits = self.output_layer(x)

        if return_activations:
            return logits, activations

        return logits


# =========================
# 5. 评估函数
# =========================

@torch.no_grad()
def evaluate(model, X_tensor, y_tensor, criterion):
    model.eval()

    logits = model(X_tensor)
    loss = criterion(logits, y_tensor).item()

    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

    acc = (preds == y_tensor).float().mean().item()

    return loss, acc


# =========================
# 6. ReLU 失活神经元统计
# =========================

@torch.no_grad()
def get_dead_neuron_counts(model, X_tensor):
    """
    判断失活神经元：
    如果某个 ReLU 神经元在全部训练样本上的输出都接近 0，
    就认为该神经元在当前 epoch 失活。
    """

    model.eval()

    _, activations = model(X_tensor, return_activations=True)

    dead_counts = []
    dead_masks = []

    for act in activations:
        # act shape: [样本数, 神经元数]
        dead_mask = (act <= 1e-8).all(dim=0).cpu().numpy().astype(int)
        dead_masks.append(dead_mask)
        dead_counts.append(int(dead_mask.sum()))

    return dead_counts, dead_masks


# =========================
# 7. 训练函数
# =========================

def train_model(
    activation_name,
    lr,
    train_loader,
    X_train_tensor,
    y_train_tensor,
    X_test_tensor,
    y_test_tensor,
    monitor_dead=False
):
    model = TitanicANN(
        input_dim=X_train_tensor.shape[1],
        activation_name=activation_name
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
        "dead_counts": [],
        "dead_masks": []
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()

        train_loss, train_acc = evaluate(model, X_train_tensor, y_train_tensor, criterion)
        test_loss, test_acc = evaluate(model, X_test_tensor, y_test_tensor, criterion)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        if monitor_dead:
            dead_counts, dead_masks = get_dead_neuron_counts(model, X_train_tensor)
            history["dead_counts"].append(dead_counts)
            history["dead_masks"].append(dead_masks)

        if epoch == 1 or epoch % 20 == 0:
            print(
                f"{activation_name:10s} | "
                f"lr={lr:<8g} | "
                f"epoch={epoch:03d} | "
                f"train_loss={train_loss:.4f} | "
                f"test_loss={test_loss:.4f} | "
                f"train_acc={train_acc:.4f} | "
                f"test_acc={test_acc:.4f}"
            )

    return model, history


# =========================
# 8. 绘图函数
# =========================

def plot_loss_curves(results):
    plt.figure(figsize=(9, 6))

    for act_name, history in results.items():
        plt.plot(history["train_loss"], label=f"{act_name} train loss")
        plt.plot(history["test_loss"], linestyle="--", label=f"{act_name} test loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves of Different Activation Functions")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, "ann_activation_loss_curves.png")
    plt.savefig(save_path, dpi=300)
    print("loss 曲线已保存：", save_path)
    plt.show()


def plot_accuracy_bar(results):
    names = list(results.keys())
    train_accs = [results[name]["train_acc"][-1] for name in names]
    test_accs = [results[name]["test_acc"][-1] for name in names]

    x = np.arange(len(names))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, train_accs, width, label="Train Acc")
    plt.bar(x + width / 2, test_accs, width, label="Test Acc")

    plt.xticks(x, names)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Final Accuracy of Different Activation Functions")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, "ann_activation_accuracy.png")
    plt.savefig(save_path, dpi=300)
    print("准确率图已保存：", save_path)
    plt.show()


def plot_dead_neuron_total(relu_results):
    plt.figure(figsize=(9, 6))

    for lr, history in relu_results.items():
        dead_counts = np.array(history["dead_counts"])

        if len(dead_counts) == 0:
            continue

        total_dead = dead_counts.sum(axis=1)

        plt.plot(
            range(1, len(total_dead) + 1),
            total_dead,
            label=f"lr={lr:g}"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Total Dead Neurons")
    plt.title("Dead ReLU Neurons under Different Learning Rates")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, "relu_dead_neurons_total.png")
    plt.savefig(save_path, dpi=300)
    print("ReLU 失活神经元数量图已保存：", save_path)
    plt.show()


def plot_dead_neuron_heatmap(history, lr):
    """
    热力图：
    灰色：没有失活；
    红色：失活。
    """

    if len(history["dead_masks"]) == 0:
        return

    cmap = ListedColormap(["lightgray", "red"])

    fig, axes = plt.subplots(len(HIDDEN_DIMS), 1, figsize=(10, 10))

    if len(HIDDEN_DIMS) == 1:
        axes = [axes]

    for layer_idx, ax in enumerate(axes):
        matrix = np.stack(
            [epoch_masks[layer_idx] for epoch_masks in history["dead_masks"]],
            axis=0
        )

        ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f"Hidden Layer {layer_idx + 1}")
        ax.set_xlabel("Neuron Index")
        ax.set_ylabel("Epoch")

    fig.suptitle(f"Dead ReLU Neuron Heatmap | lr={lr:g}")
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, f"relu_dead_heatmap_lr_{lr:g}.png")
    plt.savefig(save_path, dpi=300)
    print("ReLU 失活神经元热力图已保存：", save_path)
    plt.show()


def save_summary(results, relu_results):
    save_path = os.path.join(RESULTS_DIR, "ann_extra2_result_summary.txt")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("ANN 附加题实验结果汇总\n")
        f.write("=" * 50 + "\n\n")

        f.write("一、不同激活函数对比\n")
        for act_name, history in results.items():
            f.write(
                f"{act_name}: "
                f"train_acc={history['train_acc'][-1]:.4f}, "
                f"test_acc={history['test_acc'][-1]:.4f}, "
                f"train_loss={history['train_loss'][-1]:.4f}, "
                f"test_loss={history['test_loss'][-1]:.4f}\n"
            )

        f.write("\n二、ReLU 不同学习率下的失活神经元数量\n")
        for lr, history in relu_results.items():
            final_dead = np.array(history["dead_counts"][-1])
            f.write(
                f"lr={lr:g}: "
                f"各层失活数量={final_dead.tolist()}, "
                f"总失活数量={int(final_dead.sum())}\n"
            )

    print("实验结果汇总已保存：", save_path)


# =========================
# 9. 主程序
# =========================

def main():
    set_seed(SEED)
    make_dirs()

    print("当前设备：", DEVICE)
    print("数据路径：", DATA_PATH)

    df = read_csv_safely(DATA_PATH)
    X, y = preprocess_titanic(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y
    )

    # 归一化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # ==================================================
    # 实验一：固定学习率，不同激活函数
    # ==================================================

    print("\n========== 实验一：不同激活函数对比 ==========\n")

    activation_names = ["sigmoid", "tanh", "relu", "leaky_relu"]
    activation_results = {}

    for act_name in activation_names:
        print(f"\n开始训练激活函数：{act_name}")

        model, history = train_model(
            activation_name=act_name,
            lr=BASE_LR,
            train_loader=train_loader,
            X_train_tensor=X_train_tensor,
            y_train_tensor=y_train_tensor,
            X_test_tensor=X_test_tensor,
            y_test_tensor=y_test_tensor,
            monitor_dead=False
        )

        activation_results[act_name] = history

    print("\n========== 不同激活函数最终结果 ==========")

    for act_name, history in activation_results.items():
        print(
            f"{act_name:10s} | "
            f"train_acc={history['train_acc'][-1]:.4f} | "
            f"test_acc={history['test_acc'][-1]:.4f} | "
            f"train_loss={history['train_loss'][-1]:.4f} | "
            f"test_loss={history['test_loss'][-1]:.4f}"
        )

    plot_loss_curves(activation_results)
    plot_accuracy_bar(activation_results)

    # ==================================================
    # 实验二：ReLU 学习率扩大 10 倍、100 倍
    # ==================================================

    print("\n========== 实验二：ReLU 失活神经元观察 ==========\n")

    relu_lrs = [BASE_LR, BASE_LR * 10, BASE_LR * 100]
    relu_results = {}

    for lr in relu_lrs:
        print(f"\n开始训练 ReLU，学习率 lr={lr:g}")

        model, history = train_model(
            activation_name="relu",
            lr=lr,
            train_loader=train_loader,
            X_train_tensor=X_train_tensor,
            y_train_tensor=y_train_tensor,
            X_test_tensor=X_test_tensor,
            y_test_tensor=y_test_tensor,
            monitor_dead=True
        )

        relu_results[lr] = history

        final_dead = np.array(history["dead_counts"][-1])

        print(f"lr={lr:g} 最后每层失活神经元数量：{final_dead}")
        print(f"lr={lr:g} 最后总失活神经元数量：{final_dead.sum()}")

    plot_dead_neuron_total(relu_results)

    for lr, history in relu_results.items():
        plot_dead_neuron_heatmap(history, lr)

    save_summary(activation_results, relu_results)

    print("\n实验完成。所有图片和结果已保存到 results 文件夹。")


if __name__ == "__main__":
    main()

#ReLU 失活神经元数量与学习率有关。学习率较小时，参数更新较平稳，失活神经元数量通常较少；
#学习率增大 10 倍或 100 倍后，参数更新幅度变大，模型更容易出现
#ReLU 神经元长期输出为 0 的情况，因此失活神经元数量可能增加。