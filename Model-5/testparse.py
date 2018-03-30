import csv 
import cv2 
import numpy as np 

# ******************* Enter the input of rows and cols here(below)********************
# ******************* using format x1,y1,x2,y2,Word where x1,y1 for top left and x2,y2 for bottom right********************
# ******************* Use scal_fact as scaling factor for normalisation of font size in cv2.putText Function*********************
rows = 600
cols = 600
scal_fact = 10          
img = np.zeros((rows,cols,3))
# np.zeroes()
for h in range(rows):
    for w in range(cols):
        img[h,w] = [255,255,255]
max = 0 
with open('bounding_boxes_normal.csv') as File:
    reader = csv.reader(File, delimiter=',', quotechar=',',
                        quoting=csv.QUOTE_MINIMAL)
    for row in reader:
        a = int(row[0])
        b = int(row[1])
        c = int(row[2])
        d = int(row[3])
        x = abs((c-a)*(d-b))
        if (x>max):
            max = x
print('The value of max is')
print max
with open('bounding_boxes_normal.csv') as File:
    reader = csv.reader(File, delimiter=',', quotechar=',',
                        quoting=csv.QUOTE_MINIMAL)
    for row in reader:
        a = int(row[0])
        b = int(row[1])
        c = int(row[2])
        d = int(row[3])
        e = row[4]
        x = abs((c-a)*(d-b))
        # print e
        # print a
        # print d
        cv2.putText(img, str(e),(a,d), cv2.FONT_HERSHEY_SIMPLEX, float(scal_fact*x)/float(max),(0,0,0),2)
        # cv2.imshow('image', img)
        # cv2.waitKey(0)
cv2.imshow('image', img)
cv2.waitKey(0)

