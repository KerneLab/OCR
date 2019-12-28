import cv2
import numpy as np
from numpy.linalg import norm
from functools import reduce
import matplotlib.pyplot as plt


def imshow(img, name=''):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyWindow(name)
    # plt.imshow(img), plt.title(name)
    # plt.xticks([]), plt.yticks([])
    # plt.show()


def findFurthestApartPoints(num, points):
    vects = dict([(p, np.array(p)) for p in points])
    dists = dict()
    for a in points:
        for b in points:
            d = norm(vects[a] - vects[b])
            dists[(a, b)] = d
            dists[(b, a)] = d
    apart = []  # 分离点
    cache = dict()  # key点到其余点的距离和
    for point in points:
        if len(apart) < num:
            apart.append(point)
            if len(apart) == num:
                for p in apart:
                    cache[p] = sum([dists[(p, q)] for q in apart if p != q])
        else:
            temp = [sum([dists[(point, q)] for q in apart if p != q]) for p in apart]
            # TODO 找出离当前节点a最近的现有b，如果a与其余点的距离和大于b到其余点的距离，则用a替换b
            smax = max(temp)
            index = temp.index(smax)
            del cache[apart[index]]
            apart[index] = point
            cache[point] = smax
    return apart


# img_file = '../../img/20191228.png'
img_file = '../../img/201912082029.png'
img_origin = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
# imshow(img_origin, '')
(height, width) = img_origin.shape

img_inverse = cv2.bitwise_not(img_origin)

img_adapt = cv2.adaptiveThreshold(img_inverse, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
imshow(img_adapt)

img_depict = img_adapt.copy()
rawLines = cv2.HoughLinesP(img_depict, 1, np.pi / 180, 200, minLineLength=30, maxLineGap=10)
rawLineEnds = [line[0] for line in rawLines]

# 定位最大边界的上下左右范围
leftX, topY, rightX, bottomY = reduce(lambda a, b: [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])],
                                      rawLineEnds)

boundThick = 2

leftLines = [end for end in rawLineEnds if abs(end[0] - leftX) <= boundThick] + \
            [end for end in rawLineEnds if abs(end[2] - leftX) <= boundThick]

topLines = [end for end in rawLineEnds if abs(end[1] - topY) <= boundThick] + \
           [end for end in rawLineEnds if abs(end[3] - topY) <= boundThick]

rightLines = [end for end in rawLineEnds if abs(end[0] - rightX) <= boundThick] + \
             [end for end in rawLineEnds if abs(end[2] - rightX) <= boundThick]

bottomLines = [end for end in rawLineEnds if abs(end[1] - bottomY) <= boundThick] + \
              [end for end in rawLineEnds if abs(end[3] - bottomY) <= boundThick]

lineEnds = list(set([(end[0], end[1]) for end in leftLines] + [(end[2], end[3]) for end in leftLines] +
                    [(end[0], end[1]) for end in topLines] + [(end[2], end[3]) for end in topLines] +
                    [(end[0], end[1]) for end in rightLines] + [(end[2], end[3]) for end in rightLines] +
                    [(end[0], end[1]) for end in bottomLines] + [(end[2], end[3]) for end in bottomLines]))

print(findFurthestApartPoints(4, lineEnds))

img_frames = np.zeros((height, width), np.uint8)
for line in leftLines + topLines + rightLines + bottomLines:
    x1, y1, x2, y2 = line
    cv2.line(img_frames, (x1, y1), (x2, y2), (255, 255, 255), 1)

imshow(img_frames)
cv2.imwrite('../../img/frame.png', img_frames)

scale = 10.0

horizontal = img_depict.copy()
# imshow(horizontal)
horizontalSize = int(width / scale)
horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalSize, 1))
horizontal = cv2.erode(horizontal, horizontalStructure)
# imshow(horizontal)
horizontal = cv2.dilate(horizontal, horizontalStructure)
# imshow(horizontal)

vertical = img_depict.copy()
# imshow(vertical)
verticalSize = int(height / scale)
verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalSize))
vertical = cv2.erode(vertical, verticalStructure)
# imshow(vertical)
vertical = cv2.dilate(vertical, verticalStructure)
# imshow(vertical)

# imshow(cv2.bitwise_or(horizontal, vertical))
cv2.imwrite('../../img/yyy.png', cv2.bitwise_or(horizontal, vertical))

# imshow(cv2.bitwise_and(horizontal, vertical))
cv2.imwrite('../../img/zzz.png', cv2.bitwise_and(horizontal, vertical))
