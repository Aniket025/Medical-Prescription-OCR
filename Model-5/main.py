import preprocess as pp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from ocr.helpers import implt, resize, ratio
import word_detection as wd
import glob
import word_classification as wc
import os

image = cv2.cvtColor(cv2.imread("4.jpg"), cv2.COLOR_BGR2RGB)

#newImage = image
## Preprocess image

# Edge detection ()
imageEdges = pp.edgesDet(image, 200, 250)

# Close gaps between edges (double page clouse => rectangle kernel)
closedEdges = cv2.morphologyEx(imageEdges, cv2.MORPH_CLOSE, np.ones((5, 11)))


pageContour = pp.findPageContours(closedEdges, resize(image))

# Recalculate to original scale
pageContour = pageContour.dot(ratio(image))


newImage = pp.perspImageTransform(image, pageContour)

# cv2.imwrite("2_1.jpg", cv2.cvtColor(newImage, cv2.COLOR_BGR2RGB))


## Detect words and bounding boxes


# Image pre-processing - blur, edges, threshold, closing
blurred = cv2.GaussianBlur(newImage, (5, 5), 18)
edgeImg = wd.edgeDetect(blurred)
ret, edgeImg = cv2.threshold(edgeImg, 50, 255, cv2.THRESH_BINARY)
bwImage = cv2.morphologyEx(edgeImg, cv2.MORPH_CLOSE, np.ones((20,20), np.uint8))

bBoxes1 = wd.textDetect(bwImage, newImage)

## TO DO
## Reject bounding boxes with large size

# or

wbBoxes = wd.textDetectWatershed(bwImage, newImage)



file = open("./output/words/watershed/detection.txt","w")
list = glob.glob("./output/words/watershed/*.jpg")
for i in range(0,len(list)):
    image_filename = list[i]
    img = cv2.imread(image_filename)
    img = wc.imageNorm(img,60,border=False,tilt=True,hystNorm=True)
    img = cv2.copyMakeBorder(img,0,0,30,30,cv2.BORDER_CONSTANT, value=[0,0,0])
    gaps = wc.segmentation(img, step=2, RNN=True, debug=False)
    chars = []
    print image_filename + "\t",
    for i in range(0,(len(gaps)-1)):
        char = img[:,gaps[i]:gaps[i+1]]
        char, dim = wc.letterNorm(char, is_thresh=True, dim=True)
        if dim[0]>4 and dim[1]>4:
            chars.append(char.flatten())


    chars = np.array(chars)
    word = ''

    if len(chars) != 0:
        pred = wc.charClass.run(chars)
        for c in pred:
            word += wc.idx2char(c)

    print word

#for files in list:
#    os.remove(files)
