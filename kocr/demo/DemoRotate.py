import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot(bgr):
    b, g, r = cv2.split(bgr)  # 拆分通道
    rgb = cv2.merge([r, g, b])  # 通道的融合
    plt.imshow(rgb), plt.title('img_rgb_plt')
    plt.xticks([]), plt.yticks([])
    plt.show()


# 读取原始图片
img = cv2.imread('../../img/rawRotate.jpg')
plot(img)

# 这里我们通过人工的方式读出四个角点A，B，C，D
target_points = [[278, 189], [758, 336], [570, 1034], [65, 900]]

height = img.shape[0]
width = img.shape[1]

# four_points = np.array(((0, 0),
#                        (width - 1, 0),
#                        (width - 1, height - 1),
#                        (0, height - 1)),
#                       np.float32)
four_points = np.array(((100, 100),
                        (width - 100, 100),
                        (width - 100, height - 100),
                        (100, height - 100)),
                       np.float32)

# 统一格式
target_points = np.array(target_points, np.float32)

M = cv2.getPerspectiveTransform(target_points, four_points)

Rotated = cv2.warpPerspective(img, M, (width, height))
plot(Rotated)

cv2.imwrite("../../img/Rotated.jpg", Rotated)
