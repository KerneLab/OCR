import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

imgid = '201912082029.png'
image = cv.imread('../../img/' + imgid, cv.IMREAD_GRAYSCALE)
edges = cv.Canny(image, 100, 200, apertureSize=3)  # apertureSize是sobel算子大小，只能为1,3,5，7
cv.imshow('', edges)
cv.waitKey(0)
cv.destroyAllWindows()

# 函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
for line in lines:
    print(type(line))  # 多维数组
    x1, y1, x2, y2 = line[0]
    cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv.imshow("image line", image)
cv.waitKey(0)
cv.destroyAllWindows()
