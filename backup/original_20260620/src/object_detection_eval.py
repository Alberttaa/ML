import os
import re

# =========================
# IoU计算
# =========================
def calculate_iou(box1, box2):

    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])

    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (
        (x_right - x_left)
        * (y_bottom - y_top)
    )

    area1 = (
        (box1[2] - box1[0])
        * (box1[3] - box1[1])
    )

    area2 = (
        (box2[2] - box2[0])
        * (box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    return intersection / union


# =========================
# 提取数字
# =========================
def extract_numbers(line):

    nums = re.findall(
        r"-?\d+\.?\d*",
        line
    )

    result = []

    for n in nums:

        if "." in n:
            result.append(float(n))
        else:
            result.append(int(n))

    return result


# =========================
# 读取GT
# =========================
def load_gt(filename):

    gt_boxes = []

    with open(
        filename,
        "r",
        encoding="utf-8"
    ) as f:

        lines = f.readlines()

    for line in lines:

        nums = extract_numbers(line)

        if len(nums) == 4:
            gt_boxes.append(nums)

    return gt_boxes


# =========================
# 读取PD
# =========================
def load_pd(filename):

    pd_boxes = []

    with open(
        filename,
        "r",
        encoding="utf-8"
    ) as f:

        lines = f.readlines()

    for line in lines:

        nums = extract_numbers(line)

        if len(nums) == 5:
            pd_boxes.append(nums)

    return pd_boxes


# =========================
# 主程序
# =========================
def main():

    gt_file = r"E:\Ling\homework\aaa\My project\data\iou\gt.txt"

    pd_file = r"E:\Ling\homework\aaa\My project\data\iou\pd.txt"

    gt_boxes = load_gt(gt_file)
    pd_boxes = load_pd(pd_file)

    print("=" * 60)
    print("GT数量:", len(gt_boxes))
    print("PD数量:", len(pd_boxes))
    print("=" * 60)

    matched_gt = [False] * len(gt_boxes)

    TP = 0
    FP = 0

    print("\n【PD结果】")
    print("-" * 60)

    for idx, pd_box in enumerate(pd_boxes):

        pred = pd_box[:4]

        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(gt_boxes):

            iou = calculate_iou(
                pred,
                gt_box
            )

            if iou > best_iou:

                best_iou = iou
                best_gt_idx = gt_idx

        if (
            best_iou >= 0.5
            and not matched_gt[best_gt_idx]
        ):

            TP += 1
            matched_gt[best_gt_idx] = True

            result = "TP"

        else:

            FP += 1
            result = "FP"

        print(
            f"PD{idx+1}: "
            f"IoU={best_iou:.4f} "
            f"==> {result}"
        )

    FN = 0

    print("\n【GT结果】")
    print("-" * 60)

    for i in range(len(gt_boxes)):

        if matched_gt[i]:

            print(
                f"GT{i+1}: 已匹配"
            )

        else:

            print(
                f"GT{i+1}: FN"
            )

            FN += 1

    precision = (
        TP / (TP + FP)
        if (TP + FP) > 0
        else 0
    )

    recall = (
        TP / (TP + FN)
        if (TP + FN) > 0
        else 0
    )

    print("\n" + "=" * 60)
    print("统计结果")
    print("=" * 60)

    print("TP =", TP)
    print("FP =", FP)
    print("FN =", FN)

    print()

    print(
        f"Precision = {precision:.4f}"
    )

    print(
        f"Recall    = {recall:.4f}"
    )


if __name__ == "__main__":
    main()