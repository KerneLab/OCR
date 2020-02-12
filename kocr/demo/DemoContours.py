from functools import reduce

import cv2
import numpy as np
from numpy.linalg import norm

from kocr import basis
from kocr import FurthestApartPointsFinder
from kocr import ConvexPointsConnector


def imshow(img, name=''):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyWindow(name)
    # plt.imshow(img), plt.title(name)
    # plt.xticks([]), plt.yticks([])
    # plt.show()


# img_file = '../../img/20191228.png'
img_file = '../../img/20200101.png'
# img_file = '../../img/20200105.png'
# img_file = '../../img/201912082029.png'
# img_file = '../../img/201912182116.png'
img_origin = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
# imshow(img_origin, '')
(height, width) = img_origin.shape

img_inverse = cv2.bitwise_not(img_origin)

img_adapt = cv2.adaptiveThreshold(img_inverse, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
imshow(img_adapt)

img_depict_raw = img_adapt.copy()
rawLines = cv2.HoughLinesP(img_depict_raw, 1, np.pi / 180, 200, minLineLength=30, maxLineGap=10)
rawLineEnds = [line[0] for line in rawLines]

# 定位最大边界的上下左右范围
leftX, topY, rightX, bottomY = reduce(lambda a, b: [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])],
                                      rawLineEnds)

# 获取与边界范围沾边的直线，即直线的一端在边界上
boundThick = 5
leftLines = [end for end in rawLineEnds if abs(end[0] - leftX) <= boundThick] + \
            [end for end in rawLineEnds if abs(end[2] - leftX) <= boundThick]
topLines = [end for end in rawLineEnds if abs(end[1] - topY) <= boundThick] + \
           [end for end in rawLineEnds if abs(end[3] - topY) <= boundThick]
rightLines = [end for end in rawLineEnds if abs(end[0] - rightX) <= boundThick] + \
             [end for end in rawLineEnds if abs(end[2] - rightX) <= boundThick]
bottomLines = [end for end in rawLineEnds if abs(end[1] - bottomY) <= boundThick] + \
              [end for end in rawLineEnds if abs(end[3] - bottomY) <= boundThick]
lineEndsPoint = list(set([(end[0], end[1]) for end in leftLines] + [(end[2], end[3]) for end in leftLines] +
                         [(end[0], end[1]) for end in topLines] + [(end[2], end[3]) for end in topLines] +
                         [(end[0], end[1]) for end in rightLines] + [(end[2], end[3]) for end in rightLines] +
                         [(end[0], end[1]) for end in bottomLines] + [(end[2], end[3]) for end in bottomLines]))

aparts = FurthestApartPointsFinder.find(4, lineEndsPoint)
corners = ConvexPointsConnector.connect(aparts)

img_frames = np.zeros((height, width, 3), np.uint8)
for line in leftLines + topLines + rightLines + bottomLines:
    x1, y1, x2, y2 = line
    cv2.line(img_frames, (x1, y1), (x2, y2), (255, 255, 255), 1)

for apart, color in dict(zip(corners, [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0)])).items():
    cv2.circle(img_frames, apart, 5, color, 2)

imshow(img_frames)
cv2.imwrite('../../img/frame.png', img_frames)

redress_height, redress_width = ConvexPointsConnector.rect_shape(corners)  # 矫正后的图像尺寸
origin_corners = [(0, 0), (width, 0), (width, height), (0, height)]  # 源图像的矫正点
redress_corners = [(0, 0), (redress_width, 0), (redress_width, redress_height), (0, redress_height)]  # 矫正图像的矫正点
corners_map = dict(zip(origin_corners, redress_corners))
# 获取源角点对应的源图中的角点，再映射成矫正图像中的角点
redress = [corners_map[red] for red in ConvexPointsConnector.redress_rect_corners(corners, origin_corners)]
margin = 5
redress = list(map(lambda p: (p[0] + margin, p[1] + margin), redress))

redress_transform = cv2.getPerspectiveTransform(np.array(corners, np.float32), np.array(redress, np.float32))

real_height = int(redress_height + 2 * margin)
real_width = int(redress_width + 2 * margin)
img_redress = cv2.warpPerspective(img_adapt.copy(), redress_transform, (real_width, real_height))
img_redress_origin = cv2.warpPerspective(img_origin.copy(), redress_transform, (real_width, real_height))
# imshow(img_redress_origin)
cv2.imwrite('../../img/redress.png', img_redress_origin)

img_redress_inverse = cv2.bitwise_not(img_redress_origin)
img_redress_adapt = cv2.adaptiveThreshold(img_redress_inverse, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15,
                                          -2)

horizontal = img_redress_adapt.copy()
horizontal_scale = 10.0
# imshow(horizontal)
horizontalSize = int(width / horizontal_scale)
horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalSize, 1))
horizontal = cv2.erode(horizontal, horizontalStructure)
# imshow(horizontal)
horizontal = cv2.dilate(horizontal, horizontalStructure)
# imshow(horizontal)

vertical = img_redress_adapt.copy()
vertical_scale = 10.0
# imshow(vertical)
verticalSize = int(height / vertical_scale)
verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalSize))
vertical = cv2.erode(vertical, verticalStructure)
# imshow(vertical)
vertical = cv2.dilate(vertical, verticalStructure)
# imshow(vertical)

imshow(cv2.bitwise_or(horizontal, vertical))
cv2.imwrite('../../img/bit_or.png', cv2.bitwise_or(horizontal, vertical))
# imshow(cv2.bitwise_and(horizontal, vertical))
# TODO 交叉点 与 角点 取并集
cv2.imwrite('../../img/bit_and.png', cv2.bitwise_and(horizontal, vertical))

cross_point = cv2.bitwise_and(horizontal, vertical)
cross_ys, cross_xs = np.where(cross_point > 0)
img_cross = img_redress_adapt.copy()
for k, v in basis.clustering_points(zip(cross_xs, cross_ys), 5).items():
    print(k)
    cv2.circle(img_cross, k, 5, (255, 255, 255))
# imshow(img_cross)
cv2.imwrite('../../img/cross.png', img_cross)

# split_xs = []
# cross_sort_xs = np.sort(cross_xs)
# i = 0
# for j in range(len(cross_sort_xs) - 1):
#     if cross_sort_xs[j + 1] - cross_sort_xs[j] > 10:
#         split_xs.append(cross_sort_xs[j])
#     i = i + 1
# split_xs.append(cross_sort_xs[i])
#
# split_ys = []
# cross_sort_ys = np.sort(cross_ys)
# i = 0
# for j in range(len(cross_sort_ys) - 1):
#     if cross_sort_ys[j + 1] - cross_sort_ys[j] > 10:
#         split_ys.append(cross_sort_ys[j])
#     i = i + 1
# split_ys.append(cross_sort_ys[i])
#
# for x, y in zip(split_xs, split_ys):
#     print(x, y)
