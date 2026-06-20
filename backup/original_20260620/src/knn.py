import os
import time
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# =========================
# 路径设置
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "data", "titanic", "titanic_train.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "titanic", "titanic_test.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

TARGET_COL = "2urvived"


# =========================
# 1. 数据预处理
# =========================
def preprocess_for_knn(train_df, test_df, target_col):
    train_df = train_df.copy()
    test_df = test_df.copy()

    feature_cols = [col for col in train_df.columns if col != target_col]

    # 缺失值填补：用训练集对应列的中位数
    for col in feature_cols:
        median_value = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_value)
        test_df[col] = test_df[col].fillna(median_value)

    X_train = train_df[feature_cols].astype(float)
    X_test = test_df[feature_cols].astype(float)
    y_train = train_df[target_col].astype(int).values
    y_test = test_df[target_col].astype(int).values

    # 标准化：只用训练集统计量
    mean = X_train.mean()
    std = X_train.std().replace(0, 1)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train.values, X_test.values, y_train, y_test, feature_cols, mean, std


# =========================
# 2. KNN预测
# =========================
def knn_predict_vectorized(X_train, y_train, X_test, k=5):
    train_sq = np.sum(X_train ** 2, axis=1)
    test_sq = np.sum(X_test ** 2, axis=1).reshape(-1, 1)

    dist_sq = test_sq + train_sq - 2 * X_test @ X_train.T
    dist_sq = np.maximum(dist_sq, 0)
    distances = np.sqrt(dist_sq)

    nearest_indices = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    nearest_labels = y_train[nearest_indices]

    votes_for_1 = np.sum(nearest_labels == 1, axis=1)
    votes_for_0 = k - votes_for_1
    predictions = np.where(votes_for_1 >= votes_for_0, 1, 0)

    return predictions


def evaluate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


# =========================
# 3. KD-tree相关
# =========================
class KDNode:
    def __init__(self, point=None, label=None, axis=None, left=None, right=None, index=None):
        self.point = point
        self.label = label
        self.axis = axis
        self.left = left
        self.right = right
        self.index = index


def build_kdtree(data_with_label, depth=0):
    if len(data_with_label) == 0:
        return None

    k_dim = len(data_with_label[0][0])
    axis = depth % k_dim

    data_with_label = sorted(data_with_label, key=lambda x: x[0][axis])
    median = len(data_with_label) // 2
    point, label, index = data_with_label[median]

    left_subtree = build_kdtree(data_with_label[:median], depth + 1)
    right_subtree = build_kdtree(data_with_label[median + 1:], depth + 1)

    return KDNode(
        point=point,
        label=label,
        axis=axis,
        left=left_subtree,
        right=right_subtree,
        index=index
    )


def print_kdtree(node, kd_features, depth=0, max_depth=3):
    if node is None or depth > max_depth:
        return

    axis_name = kd_features[node.axis]
    indent = "  " * depth
    print(f"{indent}深度={depth}, 切分轴={axis_name}, 点={node.point}, 标签={node.label}, 原下标={node.index}")

    print_kdtree(node.left, kd_features, depth + 1, max_depth)
    print_kdtree(node.right, kd_features, depth + 1, max_depth)


# =========================
# 4. 主流程
# =========================
def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("正在读取 Titanic KNN 数据...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print("训练集形状：", train_df.shape)
    print("测试集形状：", test_df.shape)
    print("训练集列名：", train_df.columns.tolist())

    X_train, X_test, y_train, y_test, feature_cols, train_mean, train_std = preprocess_for_knn(
        train_df, test_df, TARGET_COL
    )

    print("\nKNN 使用的特征数：", len(feature_cols))
    print("部分特征名：", feature_cols[:10])

    print("\n========== 手写 KNN 结果 ==========\n")

    results = []
    k_values = []
    accuracies = []

    for k in range(3, 11):
        start_time = time.time()
        y_pred_vec = knn_predict_vectorized(X_train, y_train, X_test, k=k)
        vec_time = time.time() - start_time
        vec_acc = evaluate_accuracy(y_test, y_pred_vec)

        results.append({
            "k": k,
            "vectorized_accuracy": vec_acc,
            "vectorized_time": vec_time
        })

        k_values.append(k)
        accuracies.append(vec_acc)

        print(f"k = {k}, 测试准确率 = {vec_acc:.4f}, 耗时 = {vec_time:.4f} 秒")

    best_index = np.argmax(accuracies)
    best_k = k_values[best_index]
    best_acc = accuracies[best_index]

    print("\n最优 k 值：", best_k)
    print("最高测试准确率：", round(best_acc, 4))

    results_df = pd.DataFrame(results)
    print("\n各个 k 的结果：")
    print(results_df)

    # 保存 KNN 各 k 值结果
    knn_result_csv = os.path.join(RESULTS_DIR, "knn_k_results.csv")
    results_df.to_csv(knn_result_csv, index=False, encoding="utf-8-sig")
    print(f"\nKNN 各 k 值结果已保存到：{knn_result_csv}")

    # 画图并保存
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, accuracies, marker="o")
    plt.xticks(k_values)
    plt.xlabel("k值")
    plt.ylabel("测试准确率")
    plt.title("K值对KNN测试准确率的影响")
    plt.grid(True)

    plt.scatter(best_k, best_acc, s=80)
    plt.text(best_k, best_acc, f"  best k={best_k}\n  acc={best_acc:.4f}")

    plt.tight_layout()
    knn_plot_path = os.path.join(RESULTS_DIR, "knn_accuracy_curve.png")
    plt.savefig(knn_plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"KNN 准确率曲线已保存到：{knn_plot_path}")

    # =========================
    # KD-tree 构造与测试
    # =========================
    print("\n========== KD-tree 构造 ==========\n")

    kd_features = ["Age", "Fare"]

    train_kd_df = train_df[[*kd_features, TARGET_COL]].copy()
    test_kd_df = test_df[[*kd_features, TARGET_COL]].copy()

    for col in kd_features:
        median_value = train_kd_df[col].median()
        train_kd_df[col] = train_kd_df[col].fillna(median_value)
        test_kd_df[col] = test_kd_df[col].fillna(median_value)

    kd_mean = train_kd_df[kd_features].mean()
    kd_std = train_kd_df[kd_features].std().replace(0, 1)

    train_kd_df[kd_features] = (train_kd_df[kd_features] - kd_mean) / kd_std
    test_kd_df[kd_features] = (test_kd_df[kd_features] - kd_mean) / kd_std

    X_train_kd = train_kd_df[kd_features].values
    y_train_kd = train_kd_df[TARGET_COL].astype(int).values
    X_test_kd = test_kd_df[kd_features].values
    y_test_kd = test_kd_df[TARGET_COL].astype(int).values

    train_data_for_tree = [(X_train_kd[i], y_train_kd[i], i) for i in range(len(X_train_kd))]
    kd_root = build_kdtree(train_data_for_tree, depth=0)

    print("KD-tree 前 3 层结构如下：")
    print_kdtree(kd_root, kd_features, max_depth=3)

    print("\n========== KD-tree 查询与测试 ==========\n")
    kd_tree_query = KDTree(X_train_kd, leaf_size=2)
    distances, indices = kd_tree_query.query(X_test_kd, k=5)

    predictions_kd = []

    for i in range(len(X_test_kd)):
        neighbor_indices = indices[i]
        neighbor_labels = y_train_kd[neighbor_indices]

        pred = Counter(neighbor_labels).most_common(1)[0][0]
        predictions_kd.append(pred)

        print(
            f"测试样本 {i}: 5个最近邻标签 = {neighbor_labels.tolist()}, "
            f"投票结果 = {pred}, 真实标签 = {y_test_kd[i]}"
        )

    predictions_kd = np.array(predictions_kd)
    kd_acc = evaluate_accuracy(y_test_kd, predictions_kd)

    print("\nKD-tree 测试准确率：", round(kd_acc, 4))

    kd_output_df = pd.DataFrame({
        "Age": test_kd_df["Age"],
        "Fare": test_kd_df["Fare"],
        "TrueLabel": y_test_kd,
        "PredLabel": predictions_kd
    })

    kd_result_csv = os.path.join(RESULTS_DIR, "kd_tree_predictions.csv")
    kd_output_df.to_csv(kd_result_csv, index=False, encoding="utf-8-sig")
    print(f"KD-tree 预测结果已保存到：{kd_result_csv}")

    accuracy_txt = os.path.join(RESULTS_DIR, "accuracy.txt")
    with open(accuracy_txt, "a", encoding="utf-8") as f:
        f.write("===== KNN Titanic 二分类 =====\n")
        f.write(f"最优k: {best_k}\n")
        f.write(f"最高准确率: {best_acc:.4f}\n")
        f.write(f"KD-tree准确率: {kd_acc:.4f}\n\n")

    print(f"准确率摘要已追加保存到：{accuracy_txt}")


def train():
    run()


def test():
    run()


if __name__ == "__main__":
    run()