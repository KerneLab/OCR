import numpy as np
import cv2
from PIL import Image
from crnn.crnn_torch import crnnOcr as crnnOcr


def sharpen_image(image):
    kernel_sharpen = np.array([
        [-1, -1, -1, -1, -1],
        [-1, 2, 2, 2, -1],
        [-1, 2, 8, 2, -1],
        [-1, 2, 2, 2, -1],
        [-1, -1, -1, -1, -1]
    ]) / 8.0
    return cv2.filter2D(image, -1, kernel_sharpen)


def recognize_box(image, box):
    # box = (22, 36, 246, 70)   # Left Top Right Bottom
    image = Image.fromarray(cv2.cvtColor(sharpen_image(image), cv2.COLOR_BGR2RGB))
    sub = image.crop(box)
    text = crnnOcr(sub.convert('L'))
    return text
