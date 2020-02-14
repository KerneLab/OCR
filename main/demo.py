import os
import sys
import shutil
import cv2
import numpy as np

sys.path.append(os.getcwd())
from kocr import basis
from kocr import core

base_dir = './'
# img_file_raw = base_dir + 'img/20200101.png'
img_file_raw = base_dir + 'img/201912082029.png'
# img_file_raw = base_dir + 'img/201912182116.png'
img_raw = cv2.imread(img_file_raw, cv2.IMREAD_COLOR)
# basis.imshow(img_raw)

img_nostamp = core.remove_stamp(img_raw)
# basis.imshow(img_nostamp)

img_redress = core.redress_table(img_nostamp)
basis.imshow(img_redress)

img_gray = core.prepare_gray(img_redress)
frame_h, frame_v = core.outline_frame(img_gray, 6)

img_cross = img_gray.copy()
cross_points = core.cross_points(frame_h, frame_v)
# for p in cross_points:
#     cv2.circle(img_cross, p, 5, (0, 0, 255))
# basis.imshow(img_cross)

img_frames = cv2.bitwise_or(frame_h, frame_v)
basis.imshow(img_frames)
cv2.imwrite(base_dir + "img/frames.png", img_frames)

frame_lines = core.merge_lines(img_frames, 100)
img_lines = np.zeros(img_gray.shape)
for line in frame_lines:
    cv2.line(img_lines, line[0], line[1], (255, 255, 255), thickness=1)
# basis.imshow(img_lines)
# cv2.imwrite(base_dir + "img/lines.png", img_lines)

# 与交点相关的线段，包括线段上的交点、线段两端附近的交点
cross_point_lines = dict([(point, core.point_nearby_lines(point, frame_lines, 8)) for point in cross_points])

regions_dir = img_file_raw + ".region"
if os.path.exists(regions_dir):
    shutil.rmtree(regions_dir)
os.makedirs(regions_dir)

img_region = img_redress.copy()
regions = core.split_region(cross_points, cross_point_lines)
for region in regions:
    region_id = "_".join([str(r) for r in region])
    cv2.polylines(img_region, [np.array(region).astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 0, 255),
                  thickness=1)
    img_sub, leftop = core.cut_region(img_redress, region, 3, 3)
    boxes = core.detect_text(img_sub)
    for box in boxes:
        cv2.polylines(img_sub, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 0, 255),
                      thickness=1)
        cv2.polylines(img_redress, [box[:8].astype(np.int32).reshape((-1, 1, 2)) + leftop], True, color=(0, 0, 255),
                      thickness=1)
    cv2.imwrite(regions_dir + '/' + region_id + '.png', img_sub)
cv2.imwrite(base_dir + 'img/regions.png', img_region)
cv2.imwrite(base_dir + 'img/redress.png', img_redress)

# boxes = core.detect_text(img_nostamp)
# img_boxes = img_raw.copy()
# for box in boxes:
#     cv2.polylines(img_boxes, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0),
#                   thickness=1)
# basis.imshow(img_boxes)
