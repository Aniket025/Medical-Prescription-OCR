import csv
import cv2
import numpy as np

def draw(bboxes, entities, filename,rows,cols):
    #rows = 3282

    #cols = 2592
    scal_fact = 10
    img = np.zeros((rows,cols,3))

    for h in range(rows):
        for w in range(cols):
            img[h,w] = [255,255,255]

    max = 0
    for i in bboxes:
        a = i[0]['x']
        b = i[0]['y']
        c = i[2]['x']
        d = i[2]['y']
        x = abs((c-a)*(d-b))
        if (x>max):
            max = x
    for i in range(len(bboxes)):
        a = bboxes[i][0]['x']
        b = bboxes[i][0]['y']
        c = bboxes[i][2]['x']
        d = bboxes[i][2]['y']

        e = entities[i]
        x = abs((c-a)*(d-b))
        #float(scal_fact*x)/float(max)
        cv2.putText(img, str(e),(a,d), cv2.FONT_HERSHEY_SIMPLEX,2 ,(0,0,0),2)
    cv2.imwrite(filename, img)

if __name__ == "__main__":
    print "hello"
