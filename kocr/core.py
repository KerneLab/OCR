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


def cross_points(frame_h, frame_v):
    # TODO 交叉点 与 角点 取并集
    cross_point = cv2.bitwise_and(frame_h, frame_v)
    cross_ys, cross_xs = np.where(cross_point > 0)
    return align_points(list(basis.clustering_points(zip(cross_xs, cross_ys), 5).keys()))


def cut_region(image, region, margin_v=0, margin_h=0):
    margin_x = min(int((region[2] - region[0]) / 2), margin_h)
    margin_y = min(int((region[5] - region[3]) / 2), margin_v)
    return image[(region[3] + margin_y):(region[5] - margin_y), (region[0] + margin_x):(region[2] - margin_x)]


def detect_text(img):
    return detect.detect_image(img)


def find_lines(img_color):
    img_gray = cv2.bitwise_not(cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY))
    rawLines = cv2.HoughLinesP(img_gray, 1, np.pi / 180, 200, minLineLength=30, maxLineGap=10)


def imshow(img, name=''):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyWindow(name)


def outline_frame(img_gray, horizontal_scale=10.0, vertical_scale=10.0):
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

    return horizontal, vertical


def prepare_gray(img_color):
    img_gray = cv2.bitwise_not(cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY))
    return cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)


def redress_table(img):
    return redress.redress_by_corner(img)


def remove_stamp(color_image, min_threshold=210, target_channel=2):
    cs = cv2.split(color_image)
    _, stamp = cv2.threshold(cs[target_channel], min_threshold, 255, cv2.THRESH_BINARY)
    nostamp = cv2.merge([cv2.bitwise_or(c, stamp) for c in cs])
    # _, nostamp = cv2.threshold(cs[target_channel], min_threshold, 255, cv2.THRESH_BINARY)
    return nostamp


def split_region(points):
    # 按行遍历交点
    points.sort(key=lambda p: (p[1], p[0]))

    xm = basis.groupby(points, lambda p: p[0])
    for k, v in xm.items():
        xm[k] = [p[1] for p in v]

    ym = basis.groupby(points, lambda p: p[1])
    for k, v in ym.items():
        ym[k] = [p[0] for p in v]

    ps = set(points)

    regions = []
    for point in points:
        x, y = point
        zp = None
        for v in [v for v in xm[x] if v > y]:
            for u in [u for u in ym[y] if u > x]:
                if (u, v) in ps:
                    zp = (u, v)
                    break
            if zp is not None:
                break
        if zp is not None:
            region = [x, y, u, y, u, v, x, v]
            regions.append(region)

    return regions
