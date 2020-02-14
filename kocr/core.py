import numpy as np
import cv2
from kocr import detect
from kocr import redress
from kocr import basis


def align_points(points):
    xs, ys = np.hsplit(np.array(points), 2)
    xg = basis.clustering_points([(x[0],) for x in xs.tolist()], max_gap=5)
    xm = basis.group_reverse_map(xg, lambda v: v[0], lambda k: k[0])
    yg = basis.clustering_points([(y[0],) for y in ys.tolist()], max_gap=5)
    ym = basis.group_reverse_map(yg, lambda v: v[0], lambda k: k[0])
    return [(xm[point[0]], ym[point[1]]) for point in points]


def cross_points(frame_h, frame_v):
    # TODO 交叉点 与 角点 取并集
    cross_point = cv2.bitwise_and(frame_h, frame_v)
    cross_ys, cross_xs = np.where(cross_point > 0)
    return align_points(list(basis.clustering_points(zip(cross_xs, cross_ys), 5).keys()))


def cut_region(image, region, margin_v=0, margin_h=0):
    """
    Cut sub image according to the given region,
    return the sub image and its left-top point
    """
    margin_x = max(min(int((region[2] - region[0]) / 2), margin_h), 0)
    margin_y = max(min(int((region[5] - region[3]) / 2), margin_v), 0)
    x1, x2 = region[0] + margin_x, region[2] - margin_x
    y1, y2 = region[3] + margin_y, region[5] - margin_y
    cut = image[y1:y2, x1:x2]
    return cut, (x1, y1)


def detect_text(img):
    return detect.detect_image(img)


def merge_lines(img_lines, threshold,
                min_line_length=30, max_line_gap=10):
    """
    Merge lines by ends clustering
    """
    raw_lines = cv2.HoughLinesP(img_lines, 1, np.pi / 180, threshold,
                                minLineLength=min_line_length, maxLineGap=max_line_gap)
    lines = [basis.sort([(line[0][0], line[0][1]), (line[0][2], line[0][3])]) for line in raw_lines]
    ends = set(basis.flatten(lines))
    ends_map = basis.group_reverse_map(basis.clustering_points(ends, 5))
    merged_set = set([tuple(basis.sort([ends_map[line[0]], ends_map[line[1]]])) for line in lines])
    return [(line[0], line[1]) for line in merged_set]


def outline_frame(img_gray, border_thickness,
                  horizontal_scale=10.0, vertical_scale=10.0):
    (height, width) = img_gray.shape

    dilate_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (border_thickness, border_thickness))
    erode_structure = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    # 勾勒横向直线
    horizontal = img_gray.copy()
    horizontal_size = int(width / horizontal_scale)
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(horizontal, horizontal_structure)
    horizontal = cv2.dilate(horizontal, horizontal_structure)
    horizontal = cv2.dilate(horizontal, dilate_structure)
    horizontal = cv2.erode(horizontal, erode_structure)

    # 勾勒纵向直线
    vertical = img_gray.copy()
    vertical_size = int(height / vertical_scale)
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical = cv2.erode(vertical, vertical_structure)
    vertical = cv2.dilate(vertical, vertical_structure)
    vertical = cv2.dilate(vertical, dilate_structure)
    vertical = cv2.erode(vertical, erode_structure)

    return horizontal, vertical


def point_nearby_lines(point, lines, max_dist):
    return set([line for line in lines if basis.dist_point_line(point, line) <= max_dist])


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


def split_region(cross_points, point_lines):
    # 按行遍历交点
    cross_points.sort(key=lambda p: (p[1], p[0]))

    xm = basis.groupby(cross_points, lambda p: p[0])
    for k, v in xm.items():
        xm[k] = [p[1] for p in v]

    ym = basis.groupby(cross_points, lambda p: p[1])
    for k, v in ym.items():
        ym[k] = [p[0] for p in v]

    ps = set(cross_points)

    regions = []
    for a in cross_points:
        x, y = a
        b, c, d = None, None, None
        zp = None
        for v in [v for v in xm[x] if v > y]:
            d = (x, v)
            for u in [u for u in ym[y] if u > x]:
                b = (u, y)
                c = (u, v)
                if c in ps \
                        and len(point_lines[a].intersection(point_lines[b])) > 0 \
                        and len(point_lines[b].intersection(point_lines[c])) > 0 \
                        and len(point_lines[c].intersection(point_lines[d])) > 0 \
                        and len(point_lines[d].intersection(point_lines[a])) > 0:
                    zp = c
                    break
            if zp is not None:
                break
        if zp is not None:
            region = [a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1]]
            regions.append(region)

    return regions
