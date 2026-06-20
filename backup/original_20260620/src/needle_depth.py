import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# 参数
# ==========================

IMAGE_DIR = r"E:\Ling\homework\aaa\My project\data\test"

REFERENCE_LENGTH_CM = 3.0
INSERT_PART_CM = 2.0

TEMPLATE_SIZE = 40
SEARCH_RADIUS = 80

# ==========================
# 点击选点
# ==========================

points = []

def mouse_callback(event, x, y, flags, param):

    global points

    if event == cv2.EVENT_LBUTTONDOWN:

        points.append((x, y))

        print(f"点击: ({x},{y})")

# ==========================
# 第一帧
# ==========================

img0 = cv2.imread(
    os.path.join(
        IMAGE_DIR,
        "0.jpg"
    )
)

if img0 is None:

    raise FileNotFoundError(
        "找不到0.jpg"
    )

show = img0.copy()

cv2.namedWindow("Select")

cv2.setMouseCallback(
    "Select",
    mouse_callback
)

print("依次点击：")
print("1. 针最外端 A")
print("2. 握针靠皮肤端 B")
print("3. 入针点 C")

while True:

    temp = show.copy()

    for p in points:

        cv2.circle(
            temp,
            p,
            5,
            (0,0,255),
            -1
        )

    cv2.imshow(
        "Select",
        temp
    )

    if len(points) == 3:
        break

    cv2.waitKey(10)

cv2.destroyAllWindows()

A = points[0]
B = points[1]
C = points[2]

# ==========================
# 建立比例尺
# ==========================

reference_pixel = np.linalg.norm(
    np.array(A) - np.array(B)
)

scale = (
    REFERENCE_LENGTH_CM
    / reference_pixel
)

print()
print("参考长度(pixel):",
      round(reference_pixel,2))

print("比例(cm/pixel):",
      round(scale,5))

# ==========================
# 模板，模板匹配算法
# ==========================

gray0 = cv2.cvtColor(
    img0,
    cv2.COLOR_BGR2GRAY
)

def get_template(img, pt):

    x,y = pt

    return img[
        y-TEMPLATE_SIZE:y+TEMPLATE_SIZE,
        x-TEMPLATE_SIZE:x+TEMPLATE_SIZE
    ]

template_A = get_template(
    gray0,
    A
)

template_B = get_template(
    gray0,
    B
)

# ==========================
# 跟踪
# ==========================

depths = []

current_A = A
current_B = B

files = sorted(
    [
        f
        for f in os.listdir(IMAGE_DIR)
        if f.endswith(".jpg")
    ],
    key=lambda x:
        int(
            os.path.splitext(x)[0]
        )
)

for fname in files:

    path = os.path.join(
        IMAGE_DIR,
        fname
    )

    img = cv2.imread(path)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # ----------------------
    # 跟踪A
    # ----------------------

    ax, ay = current_A

    roi = gray[
        max(0, ay-SEARCH_RADIUS):
        min(gray.shape[0],
            ay+SEARCH_RADIUS),

        max(0, ax-SEARCH_RADIUS):
        min(gray.shape[1],
            ax+SEARCH_RADIUS)
    ]

    res = cv2.matchTemplate(
        roi,
        template_A,
        cv2.TM_CCOEFF_NORMED
    )

    _, _, _, max_loc = cv2.minMaxLoc(res)

    new_A = (
        max(0, ax-SEARCH_RADIUS)
        + max_loc[0]
        + TEMPLATE_SIZE,

        max(0, ay-SEARCH_RADIUS)
        + max_loc[1]
        + TEMPLATE_SIZE
    )

    current_A = new_A

    # ----------------------
    # 跟踪B
    # ----------------------

    bx, by = current_B

    roi = gray[
        max(0, by-SEARCH_RADIUS):
        min(gray.shape[0],
            by+SEARCH_RADIUS),

        max(0, bx-SEARCH_RADIUS):
        min(gray.shape[1],
            bx+SEARCH_RADIUS)
    ]

    res = cv2.matchTemplate(
        roi,
        template_B,
        cv2.TM_CCOEFF_NORMED
    )

    _, _, _, max_loc = cv2.minMaxLoc(res)

    new_B = (
        max(0, bx-SEARCH_RADIUS)
        + max_loc[0]
        + TEMPLATE_SIZE,

        max(0, by-SEARCH_RADIUS)
        + max_loc[1]
        + TEMPLATE_SIZE
    )

    current_B = new_B

    # =====================
    # 深度计算
    # =====================

    visible_pixel = np.linalg.norm(
        np.array(current_B)
        - np.array(C)
    )

    visible_cm = (
        visible_pixel
        * scale
    )

    depth = (
        INSERT_PART_CM
        - visible_cm
    )

    depth = max(
        0,
        min(
            depth,
            INSERT_PART_CM
        )
    )

    depths.append(depth)

# ==========================
# 输出
# ==========================

print()
print("深度结果：")

for i,d in enumerate(depths):

    print(
        f"Frame {i:03d}: "
        f"{d:.3f} cm"
    )

# ==========================
# 曲线
# ==========================

plt.figure(figsize=(10,5))

plt.plot(
    depths,
    linewidth=2
)

plt.xlabel("Frame")
plt.ylabel("Depth(cm)")
plt.title(
    "Needle Insertion Depth"
)

plt.grid(True)

plt.show()