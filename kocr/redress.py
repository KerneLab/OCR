import numpy as np
import cv2
from functools import reduce
from kocr import basis
from kocr import ConvexPointsConnector
from kocr import FurthestApartPointsFinder


def redress_by_corner(img_color, threshold=500, bound_thickness=5, margin=0):
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    (height, width) = img_gray.shape
    img_depict = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    depict_lines = cv2.HoughLinesP(img_depict, 1, np.pi / 180, threshold, minLineLength=30, maxLineGap=0)
    line_ends = [line[0] for line in depict_lines]

    # img_lines = np.zeros(img_gray.shape)
    # for line in depict_lines:
    #     cv2.line(img_lines, tuple(line[0][0:2]), tuple(line[0][2:4]), (255, 255, 255), thickness=1)
    # basis.imshow(img_lines)
    # cv2.imwrite('img/lines.png', img_lines)
    # contours, hierarchy = cv2.findContours(img_depict, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img_lines, contours, -1, (255, 255, 255), 3)
    # basis.imshow(img_lines)

    # 定位最大边界的上下左右范围
    left_x, top_y, right_x, bottom_y = reduce(
        lambda a, b: [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])],
        line_ends)

    # 获取与边界范围沾边的直线，即直线的一端在边界上
    left_lines = [end for end in line_ends if abs(end[0] - left_x) <= bound_thickness] + \
                 [end for end in line_ends if abs(end[2] - left_x) <= bound_thickness]
    top_lines = [end for end in line_ends if abs(end[1] - top_y) <= bound_thickness] + \
                [end for end in line_ends if abs(end[3] - top_y) <= bound_thickness]
    right_lines = [end for end in line_ends if abs(end[0] - right_x) <= bound_thickness] + \
                  [end for end in line_ends if abs(end[2] - right_x) <= bound_thickness]
    bottom_lines = [end for end in line_ends if abs(end[1] - bottom_y) <= bound_thickness] + \
                   [end for end in line_ends if abs(end[3] - bottom_y) <= bound_thickness]
    line_end_points = list(set([(end[0], end[1]) for end in left_lines] + [(end[2], end[3]) for end in left_lines] +
                               [(end[0], end[1]) for end in top_lines] + [(end[2], end[3]) for end in top_lines] +
                               [(end[0], end[1]) for end in right_lines] + [(end[2], end[3]) for end in right_lines] +
                               [(end[0], end[1]) for end in bottom_lines] + [(end[2], end[3]) for end in bottom_lines]))

    # 寻找最远分离点作为角点
    aparts = FurthestApartPointsFinder.find(4, line_end_points)
    corners = ConvexPointsConnector.connect(aparts)

    img_frames = np.zeros((height, width, 3), np.uint8)
    for line in left_lines + top_lines + right_lines + bottom_lines:
        x1, y1, x2, y2 = line
        cv2.line(img_frames, (x1, y1), (x2, y2), (255, 255, 255), 1)

    redress_height, redress_width = ConvexPointsConnector.rect_shape(corners)  # 矫正后的图像尺寸
    origin_corners = [(0, 0), (width, 0), (width, height), (0, height)]  # 源图像的矫正点
    redress_corners = [(0, 0), (redress_width, 0), (redress_width, redress_height), (0, redress_height)]  # 矫正图像的矫正点
    corners_map = dict(zip(origin_corners, redress_corners))
    # 获取源角点对应的源图中的角点，再映射成矫正图像中的角点
    redress = [corners_map[red] for red in ConvexPointsConnector.redress_rect_corners(corners, origin_corners)]
    redress = list(map(lambda p: (p[0] + margin, p[1] + margin), redress))

    redress_transform = cv2.getPerspectiveTransform(np.array(corners, np.float32), np.array(redress, np.float32))

    real_height = int(redress_height + 2 * margin)
    real_width = int(redress_width + 2 * margin)
    img_redress_origin = cv2.warpPerspective(img_color, redress_transform, (real_width, real_height))
    return img_redress_origin
