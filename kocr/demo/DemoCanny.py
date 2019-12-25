import cv2
import matplotlib.pyplot as plt

imgid = '201912082029.png'

img = cv2.imread('../../img/' + imgid, cv2.IMREAD_GRAYSCALE)

edges = cv2.Canny(img, 100, 200)

cv2.imshow('', edges)

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
