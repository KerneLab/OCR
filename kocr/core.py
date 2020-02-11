import numpy as np
import cv2
from kocr import detect
from kocr import redress
from kocr import basis


def align_points(points):
    xs, ys = np.hsplit(np.array(points), 2)
    xm = dict([(u[0], g[0]) for g, s in basis.clustering_points([(x[0],) for x in xs.tolist()], max_gap=5).items()
               for u in s])
    ym = dict([(v[0], g[0]) for g, s in basis.clustering_points([(y[0],) for y in ys.tolist()], max_gap=5).items()
               for v in s])
    return [(xm[point[0]], ym[point[1]]) for point in points]


def cross_points(img_color, horizontal_scale=10.0, vertical_scale=10.0):
    img_gray = cv2.bitwise_not(cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY))
    (height, width) = img_gray.shape

    # 勾勒横向直线
    horizontal = img_gray.copy()
    horizontal_size = int(width / horizontal_scale)
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(horizontal, horizontal_structure)
    horizontal = cv2.dilate(horizontal, horizontal_structure)

    # 勾勒纵向直线
    vertical = img_gray.copy()
    vertical_size = int(height / vertical_scale)
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical = cv2.erode(vertical, vertical_structure)
    vertical = cv2.dilate(vertical, vertical_structure)

    # TODO 交叉点 与 角点 取并集
    cross_point = cv2.bitwise_and(horizontal, vertical)
    cross_ys, cross_xs = np.where(cross_point > 0)
    img_cross = img_gray.copy()
    return align_points(list(basis.clustering_points(zip(cross_xs, cross_ys), 5).keys()))


def detect_text(img):
    return detect.detect_image(img)


def imshow(img, name=''):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyWindow(name)


def redress_table(img):
    return redress.redress_by_corner(img)


def remove_stamp(color_image, min_threshold=210, target_channel=2):
    cs = cv2.split(color_image)
    _, stamp = cv2.threshold(cs[target_channel], min_threshold, 255, cv2.THRESH_BINARY)
    nostamp = cv2.merge([cv2.bitwise_or(c, stamp) for c in cs])
    # _, nostamp = cv2.threshold(cs[target_channel], min_threshold, 255, cv2.THRESH_BINARY)
    return nostamp
