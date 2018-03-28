import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from ocr.helpers import implt, resize
from ocr import page
from ocr import words

class Preprocess(object):
    def __init__(self, filename):
        self.path = filename

    def read_image():
        image = cv2.cvtColor(cv2.imread(self.path), cv2.COLOR_BGR2RGB)
        return image

    def crop_image(image):
        return page.detection(image)

    def binarization(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        gray = cv2.medianBlur(gray, 3)
        return gray
