import cv2
from PIL import Image
from crnn.crnn_torch import crnnOcr as crnnOcr


def recognize_box(image, box):
    # box = (22, 36, 246, 70)
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    sub = image.crop(box)
    text = crnnOcr(sub.convert('L'))
    return text
