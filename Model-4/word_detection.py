import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from ocr.helpers import implt, resize, ratio
from copy import deepcopy

#implt(img, 'gray')

def sobelDetect(channel):
    """ The Sobel Operator"""
    sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
    sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
    # Combine x, y gradient magnitudes sqrt(x^2 + y^2)
    sobel = np.hypot(sobelX, sobelY)
    sobel[sobel > 255] = 255
    return np.uint8(sobel)


def edgeDetect(im):
    """
    Edge detection
    The Sobel operator is applied for each image layer (RGB)
    """
    return np.max(np.array([sobelDetect(im[:,:, 0]), sobelDetect(im[:,:, 1]), sobelDetect(im[:,:, 2]) ]), axis=0)

#implt(edgeImg, 'gray', 'Sobel operator')
#implt(bwImage, 'gray', 'Final closing')

def delLines(gray):
    """ Delete page lines """
    linek = np.ones((1,11),np.uint8)
    x = cv2.morphologyEx(gray, cv2.MORPH_OPEN, linek ,iterations=1)
    i = gray-x
    closing = cv2.morphologyEx(dil, cv2.MORPH_CLOSE, np.ones((17,17), np.uint8))
    #implt(closing, 'gray', 'Del Lines')
    return closing


def delBigAreas(img):
    """ Find and remove contours too big for a word """
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # ret, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 101, 3)
    #implt(gray, 'gray')

    gray2 = gray.copy()
    mask = np.zeros(gray.shape,np.uint8)

    im2, contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if (200 < cv2.contourArea(cnt) < 5000):
            cv2.drawContours(img,[cnt],0,(0,255,0),2)
            cv2.drawContours(mask,[cnt],0,255,-1)

    #implt(mask)
    #implt(img)

def union(a,b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    return [x, y, w, h]

def isIntersect(a,b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0:
        return False
    return True

def groupRectangles(rec):
    """
    Uion intersecting rectangles
    Args:
        rec - list of rectangles in form [x, y, w, h]
    Return:
        list of grouped ractangles
    """
    tested = [False for i in range(len(rec))]
    final = []
    i = 0
    while i < len(rec):
        if not tested[i]:
            j = i+1
            while j < len(rec):
                if not tested[j] and isIntersect(rec[i], rec[j]):
                    rec[i] = union(rec[i], rec[j])
                    tested[j] = True
                    j = i
                j += 1
            final += [rec[i]]
        i += 1

    return final

def textDetect(img, original):
    """ Text detection using contours """
    # Resize image
    small = resize(img, 2000)
    image = resize(original, 2000)

    # Finding contours
    mask = np.zeros(small.shape, np.uint8)
    im2, cnt, hierarchy = cv2.findContours(np.copy(small), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    #implt(img, 'gray')

    # Variables for contour index and words' bounding boxes
    index = 0
    boundingBoxes = np.array([0,0,0,0])
    bBoxes = []

    # CCOMP hierarchy: [Next, Previous, First Child, Parent]
    # cv2.RETR_CCOMP - contours into 2 levels
    # Go through all contours in first level
    while (index >= 0):
        x,y,w,h = cv2.boundingRect(cnt[index])
        # Get only the contour
        cv2.drawContours(mask, cnt, index, (255, 255, 255), cv2.FILLED)
        maskROI = mask[y:y+h, x:x+w]
        # Ratio of white pixels to area of bounding rectangle
        r = float(cv2.countNonZero(maskROI)) / (w * h)

        # Limits for text (white pixel ratio, width, height)
        # TODO Test h/w and w/h ratios
        if r > 0.1 and 1600 > w > 10 and 1600 > h > 10 and h/w < 3 and w/h < 10 and (60 // h) * w < 1000:
            bBoxes += [[x, y, w, h]]

        # Index of next contour
        index = hierarchy[0][index][0]

    # Group intersecting rectangles
    bBoxes = groupRectangles(bBoxes)
    i = 0
    f = open("output/words/normal/bounding_boxes_normal.txt","w")
    for (x, y, w, h) in bBoxes:
        boundingBoxes = np.vstack((boundingBoxes, np.array([x, y, x+w, y+h])))
        cv2.imwrite("output/words/normal/"+str(i)+".jpg",image[y:y+h, x:x+w])
        f.write(str(i) + "\t => \t" + "("+str(x)+","+str(y)+")"+","+"("+str(x+w)+","+str(y+h)+")"+"\n")
        # cv2.rectangle(image, (x, y),(x+w,y+h), (0, 255, 0), 2)
        i = i+1
    #implt(image, t='Bounding rectangles')

    # Recalculate coordinates to original scale
    bBoxes = boundingBoxes.dot(ratio(image, small.shape[0])).astype(np.int64)
    return bBoxes[1:]

def textDetectWatershed(thresh, original):
    """ Text detection using watershed algorithm """
    # According to: http://docs.opencv.org/trunk/d3/db4/tutorial_py_watershed.html
    img = resize(original, 3000)
    thresh = resize(thresh, 3000)
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 3)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.01*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    # cv2.imshow("image",sure_fg)
    # cv2.waitKey(10)
   
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)



    # Add one to all labels so that sure background is not 0, but 1
    markers += 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0


 
    markers = cv2.watershed(img, markers)

    #implt(markers, t='Markers')
    image = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Creating result array
    boundingBoxes = np.array([0,0,0,0])
    bBoxes = []

    for mark in np.unique(markers):
        # mark == 0 --> background
        if mark == 0:
            continue

        # Draw it on mask and detect biggest contour
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == mark] = 255

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        c = max(cnts, key=cv2.contourArea)

        # Draw a bounding rectangle if it contains text
        x,y,w,h = cv2.boundingRect(c)
        cv2.drawContours(mask, c, 0, (255, 255, 255), cv2.FILLED)
        maskROI = mask[y:y+h, x:x+w]
        # Ratio of white pixels to area of bounding rectangle
        r = float(cv2.countNonZero(maskROI)) / (w * h)
        # Limits for text
        

        # WORK ON THIS
        if r > 0.1 and 2000 > w > 15 and 1500 > h > 15:
            bBoxes += [[x, y, w, h]]

    # Group intersecting rectangles
    #bBoxes = groupRectangles(bBoxes)
    i = 0
    f = open("output/words/watershed/bounding_boxes_watershed.txt","w")
    for (x, y, w, h) in bBoxes:
        boundingBoxes = np.vstack((boundingBoxes, np.array([x, y, x+w, y+h])))
        cv2.imwrite("output/words/watershed/"+str(i)+".jpg",image[y:y+h, x:x+w])
        f.write(str(i) + "\t => \t" + "("+str(x)+","+str(y)+")"+","+"("+str(x+w)+","+str(y+h)+")"+"\n")
        # cv2.rectangle(image, (x, y),(x+w,y+h), (0, 255, 0), 2)
        i = i+1

    #implt(image)

    # Recalculate coordinates to original size
    bBoxes = boundingBoxes.dot(ratio(original, 3000)).astype(np.int64)
    return bBoxes[1:]

#print(len(wbBoxes))
#print(len(bBoxes))

##---
# image = cv2.cvtColor(cv2.imread("2_1.jpg"), cv2.COLOR_BGR2RGB)
# img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


# # Image pre-processing - blur, edges, threshold, closing
# blurred = cv2.GaussianBlur(image, (5, 5), 18)
# edgeImg = edgeDetect(blurred)
# ret, edgeImg = cv2.threshold(edgeImg, 50, 255, cv2.THRESH_BINARY)
# bwImage = cv2.morphologyEx(edgeImg, cv2.MORPH_CLOSE, np.ones((20,20), np.uint8))



# bBoxes1 = textDetect(bwImage, image)

# # or

# wbBoxes = textDetectWatershed(bwImage, image)