import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from PIL import Image
import pytesseract
import os

IMG = '1'    # 1, 2, 3
filename = "test/3.png"
save_filename = "test/2_1.jpg"

text = pytesseract.image_to_string(Image.open("test/3.png"))
print(text)
