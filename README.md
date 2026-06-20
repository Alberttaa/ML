# ML_Project

一个按照课程统一工程框架整理的机器学习实验项目。当前版本在尽量复用你原有实验代码、数据与结果文件的前提下，补齐了统一目录、统一入口 `main.py`、标准化预处理模块、算法模块说明、`requirements.txt`、`.gitignore` 与 GitHub 展示用 `README.md`。

## 1. Project Overview

本项目聚合了以下课程实验方向：

| Algorithm | Supported Dataset | Notes |
| --- | --- | --- |
| Linear Regression | House | 房价预测，输出评估指标与预测图 |
| KNN | Titanic / MNIST / CIFAR-10 | 图像任务默认做采样，避免运行时间过长 |
| Logistic Regression | Titanic / MNIST(HOG, binary) / CIFAR-10(HOG, binary) | 图像二分类默认类别可自定义 |
| SVM | Titanic / MNIST(HOG, binary) / CIFAR-10(HOG, binary) | 与 Logistic 共用 HOG 管线 |
| ANN | House / Titanic / CIFAR-10 | CIFAR-10 使用 PyTorch MLP，输出 loss / accuracy 曲线 |

## 2. Project Structure

```text
ML_Project/
├── backup/
│   └── original_20260620/        # 原始入口与 src 备份
├── data/
│   ├── house/
│   ├── titanic/
│   ├── mnist/
│   │   ├── train/
│   │   └── test/
│   └── cifar10/
│       ├── train/
│       └── test/
├── models/
├── results/
├── src/
│   ├── ann.py
│   ├── data_preprocess.py
│   ├── hog_features.py
│   ├── knn.py
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── svm.py
│   ├── utils.py
│   └── legacy/                   # 保留的旧实验脚本
├── main.py
├── README.md
├── requirements.txt
└── .gitignore
```

## 3. Supported Datasets

| Dataset | Path | Current Format |
| --- | --- | --- |
| House | `data/house/house_data.csv` | 数值特征 + `y` |
| Titanic | `data/titanic/train.csv` / `data/titanic/test.csv` | 标签列支持 `Survived` 或 `2urvived` |
| MNIST | `data/mnist/train/<class>` / `data/mnist/test/<class>` | 文件夹名即类别标签 |
| CIFAR-10 | `data/cifar10/train/<class>` / `data/cifar10/test/<class>` | 文件夹名即类别标签 |

## 4. Environment Setup

```bash
pip install -r requirements.txt
```

建议使用独立虚拟环境，例如 `venv` 或 `.venv`。

## 5. Quick Start

```bash
python main.py --algo linear --data house --process train
python main.py --algo linear --data house --process test

python main.py --algo knn --data titanic --process train
python main.py --algo knn --data mnist --process test --max-train 2000 --max-test 400

python main.py --algo logistic --data titanic --process train
python main.py --algo logistic --data mnist --process train --class-a 0 --class-b 1
python main.py --algo logistic --data cifar10 --process test --class-a airplane --class-b automobile

python main.py --algo svm --data titanic --process train
python main.py --algo svm --data mnist --process train --class-a 3 --class-b 8

python main.py --algo ann --data house --process train
python main.py --algo ann --data titanic --process test
python main.py --algo ann --data cifar10 --process train --epochs 5 --batch-size 64
```

## 6. main.py Parameters

| Parameter | Meaning |
| --- | --- |
| `--algo` | `linear` / `knn` / `logistic` / `svm` / `ann` |
| `--data` | `house` / `titanic` / `mnist` / `cifar10` |
| `--process` | `train` / `test` / `predict` |
| `--model` | 指定模型文件路径 |
| `--result` | 指定结果输出目录 |
| `--class-a` / `--class-b` | HOG + Logistic / SVM 二分类类别 |
| `--max-train` / `--max-test` | 图像数据采样数，便于快速实验 |
| `--epochs` / `--batch-size` | ANN 图像任务训练参数 |

## 7. Result Files

训练或测试后，结果默认写入 `results/`：

- `accuracy.txt`：统一汇总主要指标
- `loss_curve_*.png`：训练损失曲线
- `accuracy_*.png`：训练准确率曲线
- `prediction_*.png`：混淆矩阵或预测图
- `sample_*.png`：图像任务样例预测
- `*_predictions_*.csv`：表格形式预测结果

模型默认写入 `models/`，命名格式为：

```text
算法_数据集_日期.pkl
算法_数据集_日期.pth
```

## 8. Legacy Files

为了不覆盖你原来的实验材料，本次整理做了两层保留：

1. `backup/original_20260620/` 保存了本次整理前的 `main.py` 与 `src/` 备份；
2. 原有的旧实验脚本会保留在 `src/legacy/` 中，方便回看和对照。

## 9. GitHub Upload Notes

推荐仓库名：`ML_Project`

建议上传的内容：

- `src/`
- `main.py`
- `README.md`
- `requirements.txt`
- `.gitignore`
- 小体积示例数据（如 `house`、`titanic`）
- 课程报告需要展示的小体积结果图

不建议直接上传的大文件：

- `data/mnist/`
- `data/cifar10/`
- `data/cifar-10-batches-py/`
- 过大的模型文件

如果确实需要上传大文件，请使用 Git LFS，而不是直接推送导致失败。

## 10. Attention

1. MNIST 与 CIFAR-10 默认规模较大，KNN/ANN 训练时建议先使用 `--max-train` 和 `--max-test` 做采样。
2. Logistic / SVM 的图像任务当前按老师要求实现为 HOG 特征二分类。
3. `predict` 目前为了统一接口，默认复用测试集流程；如果后续你需要单独读入自定义文件预测，可以在现有框架上继续扩展。
4. 如果你的 GitHub 远程仓库还没建好，请先创建仓库，再提供仓库地址用于推送。
