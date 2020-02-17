import numpy as np
import cv2
from PIL import Image
from crnn.crnn_torch import crnnOcr as crnnOcr


def enhance_contrast(img_color):
    # return cv2.normalize(img_color, dst=None, alpha=200, beta=10, norm_type=cv2.NORM_MINMAX)
    return cv2.createCLAHE(clipLimit=40, tileGridSize=(8, 8)).apply(cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY))


def sharpen_image(image):
    kernel_sharpen = np.array([
        [-1, -1, -1, -1, -1],
        [-1, 2, 2, 2, -1],
        [-1, 2, 8, 2, -1],
        [-1, 2, 2, 2, -1],
        [-1, -1, -1, -1, -1]
    ]) / 8.0
    return cv2.filter2D(image, -1, kernel_sharpen)


def refine_image(img_color):
    return sharpen_image(enhance_contrast(img_color))


def recognize_box(image, box):
    # box = (22, 36, 246, 70)   # Left Top Right Bottom
    refine = refine_image(image)
    image = Image.fromarray(refine)
    sub = image.crop(box)
    text = crnnOcr(sub.convert('L'))
    return text
