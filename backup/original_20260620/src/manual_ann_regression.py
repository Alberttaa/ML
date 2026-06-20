import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 指定Windows中文字体
font_path = "C:/Windows/Fonts/msyh.ttc"

my_font = font_manager.FontProperties(fname=font_path)

plt.rcParams["font.family"] = my_font.get_name()
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "ann", "回归.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

LEARNING_RATE = 1.0


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def mse_loss(y_pred, y_true):
    return 0.5 * (y_pred - y_true) ** 2


def init_params():
    np.random.seed(42)

    W1 = np.random.randn(4, 5) * 0.5
    b1 = np.zeros((4, 1))

    W2 = np.random.randn(3, 4) * 0.5
    b2 = np.zeros((3, 1))

    W3 = np.random.randn(1, 3) * 0.5
    b3 = np.zeros((1, 1))

    return W1, b1, W2, b2, W3, b3


def forward(x, W1, b1, W2, b2, W3, b3):
    z1 = W1 @ x + b1
    a1 = relu(z1)

    z2 = W2 @ a1 + b2
    a2 = relu(z2)

    z3 = W3 @ a2 + b3
    y_pred = z3

    cache = {
        "x": x,
        "z1": z1,
        "a1": a1,
        "z2": z2,
        "a2": a2,
        "z3": z3,
        "y_pred": y_pred
    }

    return y_pred, cache


def backward(y_true, cache, W1, W2, W3):
    x = cache["x"]
    z1 = cache["z1"]
    a1 = cache["a1"]
    z2 = cache["z2"]
    a2 = cache["a2"]
    y_pred = cache["y_pred"]

    # loss = 0.5 * (y_pred - y)^2
    dz3 = y_pred - y_true

    dW3 = dz3 @ a2.T
    db3 = dz3

    da2 = W3.T @ dz3
    dz2 = da2 * relu_derivative(z2)

    dW2 = dz2 @ a1.T
    db2 = dz2

    da1 = W2.T @ dz2
    dz1 = da1 * relu_derivative(z1)

    dW1 = dz1 @ x.T
    db1 = dz1

    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2,
        "dW3": dW3,
        "db3": db3
    }

    return grads


def update_params(W1, b1, W2, b2, W3, b3, grads):
    W1 = W1 - LEARNING_RATE * grads["dW1"]
    b1 = b1 - LEARNING_RATE * grads["db1"]

    W2 = W2 - LEARNING_RATE * grads["dW2"]
    b2 = b2 - LEARNING_RATE * grads["db2"]

    W3 = W3 - LEARNING_RATE * grads["dW3"]
    b3 = b3 - LEARNING_RATE * grads["db3"]

    return W1, b1, W2, b2, W3, b3


def mat_to_str(name, mat):
    return f"{name}:\n" + np.array2string(
        mat,
        precision=3,
        suppress_small=True
    )


def draw_network(sample_id, x, y_true, cache, loss, params_before, grads):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    W1, b1, W2, b2, W3, b3 = params_before

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off")

    input_pos = [(0.08, 0.80 - i * 0.11) for i in range(5)]
    h1_pos = [(0.28, 0.76 - i * 0.13) for i in range(4)]
    h2_pos = [(0.48, 0.70 - i * 0.16) for i in range(3)]
    out_pos = [(0.68, 0.54)]

    def draw_node(pos, value, color):
        circle = plt.Circle(pos, 0.045, color=color, ec="black", lw=1.5)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], f"{float(value):.3f}", ha="center", va="center", fontsize=12)

    ax.text(0.06, 0.90, "输入层", fontsize=16)
    ax.text(0.25, 0.90, "隐藏层1", fontsize=16)
    ax.text(0.45, 0.90, "隐藏层2", fontsize=16)
    ax.text(0.65, 0.90, "输出层", fontsize=16)

    for i, pos in enumerate(input_pos):
        draw_node(pos, x[i, 0], "#c9e6e8")

    for i, pos in enumerate(h1_pos):
        draw_node(pos, cache["a1"][i, 0], "#9bd35b")

    for i, pos in enumerate(h2_pos):
        draw_node(pos, cache["a2"][i, 0], "#9bd35b")

    draw_node(out_pos[0], cache["y_pred"][0, 0], "#ff3333")

    ax.text(0.60, 0.38, f"真实值 y = {float(y_true):.3f}", fontsize=15)
    ax.text(0.60, 0.33, f"MSE Loss = {float(loss):.6f}", fontsize=15)

    text_left = (
        mat_to_str("W1 更新前", W1) + "\n\n" +
        mat_to_str("b1 更新前", b1) + "\n\n" +
        mat_to_str("dW1 当前梯度", grads["dW1"]) + "\n\n" +
        mat_to_str("db1 当前梯度", grads["db1"])
    )

    text_mid = (
        mat_to_str("W2 更新前", W2) + "\n\n" +
        mat_to_str("b2 更新前", b2) + "\n\n" +
        mat_to_str("dW2 当前梯度", grads["dW2"]) + "\n\n" +
        mat_to_str("db2 当前梯度", grads["db2"])
    )

    text_right = (
        mat_to_str("W3 更新前", W3) + "\n\n" +
        mat_to_str("b3 更新前", b3) + "\n\n" +
        mat_to_str("dW3 当前梯度", grads["dW3"]) + "\n\n" +
        mat_to_str("db3 当前梯度", grads["db3"])
    )

    ax.text(0.02, -0.16, text_left, fontsize=8, fontproperties=my_font, va="bottom")
    ax.text(0.35, -0.16, text_mid, fontsize=8, fontproperties=my_font, va="bottom")
    ax.text(0.68, -0.16, text_right, fontsize=8, fontproperties=my_font, va="bottom")

    ax.set_title(
        f"样本 {sample_id}：前向传播 + BP反向传播 + 参数更新前梯度展示",
        fontsize=18,
        pad=20
    )

    save_path = os.path.join(RESULTS_DIR, f"manual_ann_sample_{sample_id}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"样本 {sample_id} 可视化结果已保存到：{save_path}")
##经过三层神经网络，得到预测值，和真实值计算loss##

def run():
    print("正在读取数据：", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    print("数据内容：")
    print(df)

    x_cols = ["(x_1)", "(x_2)", "(x_3)", "(x_4)", "(x_5)"]
    y_col = "(y)"

    W1, b1, W2, b2, W3, b3 = init_params()

    for idx, row in df.iterrows():
        sample_id = int(row["样本"])

        x = row[x_cols].values.astype(float).reshape(5, 1)
        y_true = float(row[y_col])

        params_before = (
            W1.copy(), b1.copy(),
            W2.copy(), b2.copy(),
            W3.copy(), b3.copy()
        )

        y_pred, cache = forward(x, W1, b1, W2, b2, W3, b3)
        loss = mse_loss(y_pred[0, 0], y_true)

        grads = backward(y_true, cache, W1, W2, W3)
##反向传播求梯度，更新参数##
        print("\n==============================")
        print(f"样本 {sample_id}")
        print("输入 x：", x.flatten())
        print("真实值 y：", y_true)
        print("预测值 y_pred：", float(y_pred[0, 0]))
        print("Loss：", float(loss))
        print(mat_to_str("W1 更新前", W1))
        print(mat_to_str("dW1 当前梯度", grads["dW1"]))

        draw_network(
            sample_id=sample_id,
            x=x,
            y_true=y_true,
            cache=cache,
            loss=loss,
            params_before=params_before,
            grads=grads
        )

        W1, b1, W2, b2, W3, b3 = update_params(
            W1, b1, W2, b2, W3, b3, grads
        )

    print("\n三组数据已经全部计算并完成参数更新。")


if __name__ == "__main__":
    run()