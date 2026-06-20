import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# =========================
# 1. 读取图片，支持中文路径
# =========================

def read_image_chinese_path(path):
    """
    cv2.imread 有时候读不了中文路径，
    所以这里用 np.fromfile + cv2.imdecode。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到图片文件：{path}")

    img_array = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("图片读取失败，请检查图片格式是否正确。")

    return img


# =========================
# 2. 添加椒盐噪声
# =========================

def add_salt_pepper_noise(img, noise_ratio=0.03):
    """
    给图片添加椒盐噪声。
    noise_ratio 越大，噪声越明显。
    """
    noisy_img = img.copy()
    h, w = noisy_img.shape[:2]

    noise_num = int(h * w * noise_ratio)

    for _ in range(noise_num):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)

        if np.random.rand() < 0.5:
            noisy_img[y, x] = [0, 0, 0]          # 椒噪声，黑点
        else:
            noisy_img[y, x] = [255, 255, 255]    # 盐噪声，白点

    return noisy_img


# =========================
# 3. SIFT 特征点匹配
# =========================

def sift_match(img1, img2, title="SIFT Match"):
    """
    输入两张图片，使用 SIFT 提取特征点，并进行特征匹配。
    """

    # 转灰度图
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 创建 SIFT 检测器
    sift = cv2.SIFT_create()

    # 检测关键点和描述子
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    print(f"\n{title}")
    print(f"原图特征点数量：{len(kp1)}")
    print(f"变换图特征点数量：{len(kp2)}")

    if des1 is None or des2 is None:
        print("没有检测到足够的特征点，无法匹配。")
        return

    # BFMatcher 暴力匹配
    bf = cv2.BFMatcher(cv2.NORM_L2)

    # knnMatch 找到每个特征点最接近的两个匹配点
    matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test，筛选较好的匹配点
    good_matches = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f"优秀匹配点数量：{len(good_matches)}")

    # 绘制前 50 个匹配点
    result_img = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        good_matches[:50],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # OpenCV 是 BGR，matplotlib 是 RGB，需要转换
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(14, 7))
    plt.imshow(result_img)
    plt.title(title)
    plt.axis("off")
    plt.show()


# =========================
# 4. 主函数
# =========================

def main():
    # 改成你自己的图片路径
    image_path = r"E:\Ling\homework\aaa\My project\data\0401.jpeg"

    # 读取原图 A
    img_A = read_image_chinese_path(image_path)

    # 为了显示和匹配更方便，可以统一调整一下原图大小
    img_A = cv2.resize(img_A, (500, 400))

    # A1：缩放图像
    img_A1 = cv2.resize(img_A, None, fx=0.6, fy=0.6)

    # A2：水平翻转图像
    img_A2 = cv2.flip(img_A, 1)

    # A3：添加椒盐噪声图像
    img_A3 = add_salt_pepper_noise(img_A, noise_ratio=0.03)

    # 显示原图和三种变换图
    show_imgs = [
        cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(img_A1, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(img_A2, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(img_A3, cv2.COLOR_BGR2RGB),
    ]

    titles = [
        "Original Image A",
        "Scaled Image A1",
        "Flipped Image A2",
        "Salt-Pepper Noise Image A3"
    ]

    plt.figure(figsize=(12, 8))

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(show_imgs[i])
        plt.title(titles[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()

    # 分别进行 SIFT 特征匹配
    sift_match(img_A, img_A1, title="SIFT Match: A and Scaled Image A1")
    sift_match(img_A, img_A2, title="SIFT Match: A and Flipped Image A2")
    sift_match(img_A, img_A3, title="SIFT Match: A and Salt-Pepper Noise Image A3")


if __name__ == "__main__":
    main()