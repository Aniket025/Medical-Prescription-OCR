import preprocess as pp
import word_detection as wd
#import word_classification as wc
import google_ocr as go
import create_prescription as cp
import spell

import argparse
import glob
import os
import numpy as np
import cv2
import sys
import time

#Running instructions - python2 main_2.py --google-ocr
parser = argparse.ArgumentParser()
parser.add_argument('--google-ocr', action="store_true",dest='google_ocr',default=False)
parser.add_argument('--file', action="store", dest="filename", type=str)

if __name__ == "__main__":
	dictionary = {}
	start_time = time.time()
	spell.create_dictionary(dictionary,"./all_medical_terms.txt")
	run_time = time.time() - start_time
	print '%.2f seconds to run' % run_time

	filename = parser.parse_args().filename
	#Load image and change channels
	image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)

	#Pre-processing = Cropping + Binarization
	imageEdges = pp.edgesDet(image, 200, 250)
	closedEdges = cv2.morphologyEx(imageEdges, cv2.MORPH_CLOSE, np.ones((5, 11)))
	pageContour = pp.findPageContours(closedEdges, pp.resize(image))
	pageContour = pageContour.dot(pp.ratio(image))
	newImage = pp.perspImageTransform(image, pageContour)

	#Saving image to show status in the app
	save_filename = filename[:-4]+"_1"+filename[-4:]
	cv2.imwrite(save_filename, cv2.cvtColor(newImage, cv2.COLOR_BGR2RGB))

	##Detect words using google-ocr
	if (parser.parse_args().google_ocr):
		entities,bBoxes = go.convert(save_filename)
		detected_filename = "input.txt"
		with open(detected_filename, 'w') as outfile:
			for i in entities:
				outfile.write(i)
				outfile.write("\n")
		spell.spellcheck(dictionary, "./input.txt")
		entities = []
		with open("output.txt") as file:
			entities = file.readlines()
		entities = [x.strip() for x in entities]
		save_filename = filename[:-4]+"_2"+filename[-4:]
		cp.draw(bBoxes,entities,save_filename,newImage.shape[0],newImage.shape[1])

	else:
		blurred = cv2.GaussianBlur(newImage, (5, 5), 18)
		edgeImg = wd.edgeDetect(blurred)
		ret, edgeImg = cv2.threshold(edgeImg, 50, 255, cv2.THRESH_BINARY)
		bwImage = cv2.morphologyEx(edgeImg, cv2.MORPH_CLOSE, np.ones((20,20), np.uint8))

		#bBoxes1 = wd.textDetect(bwImage, newImage)
		wbBoxes = wd.textDetectWatershed(bwImage, newImage)
