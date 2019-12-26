import cv2
import numpy as np
import matplotlib.pyplot as plt


def imshow(img, name=''):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyWindow(name)
    # plt.imshow(img), plt.title(name)
    # plt.xticks([]), plt.yticks([])
    # plt.show()


img_file = '../../img/201912082029.png'
img_origin = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
# imshow(img_origin, '')
(height, width) = img_origin.shape

img_inverse = cv2.bitwise_not(img_origin)

img_adapt = cv2.adaptiveThreshold(img_inverse, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
# imshow(img_adapt)

img_depict = img_adapt.copy()
lines = cv2.HoughLinesP(img_depict, 1, np.pi / 180, 100, minLineLength=30, maxLineGap=10)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img_depict, (x1, y1), (x2, y2), (255, 255, 255), 1)
imshow(img_depict)

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
imshow(vertical)
verticalSize = int(height / scale)
verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalSize))
vertical = cv2.erode(vertical, verticalStructure)
imshow(vertical)
vertical = cv2.dilate(vertical, verticalStructure)
imshow(vertical)

imshow(cv2.bitwise_or(horizontal, vertical))
cv2.imwrite('../../img/yyy.png', cv2.bitwise_or(horizontal, vertical))

# imshow(cv2.bitwise_and(horizontal, vertical))
cv2.imwrite('../../img/zzz.png', cv2.bitwise_and(horizontal, vertical))

