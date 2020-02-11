import os
import sys
import cv2
import numpy as np

sys.path.append(os.getcwd())
from kocr import core

base_dir = './'
# img_file_raw = base_dir + 'img/20200101.png'
img_file_raw = base_dir + 'img/201912082029.png'
img_raw = cv2.imread(img_file_raw, cv2.IMREAD_COLOR)
# kocr.imshow(img_raw)

img_nostamp = core.remove_stamp(img_raw)
# core.imshow(img_nostamp)

img_redress = core.redress_table(img_nostamp)
# core.imshow(img_redress)

img_cross = img_redress.copy()
cross_points = core.cross_points(img_redress)
for p in cross_points:
    print(p)
    cv2.circle(img_cross, p, 5, (0, 0, 255))
core.imshow(img_cross)

# boxes = core.detect_text(img_nostamp)
# img_boxes = img_raw.copy()
# for box in boxes:
#     cv2.polylines(img_boxes, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0),
#                   thickness=1)
# core.imshow(img_boxes)
