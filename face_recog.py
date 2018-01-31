from imutils.video import VideoStream					#importing the necessary libraries
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np
from scipy.spatial import distance as dist
import glob
import os
import sys


ap = argparse.ArgumentParser()                                    	# construct the argument parse and parse the arguments

ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
ap.add_argument("-s","--fac", required=True,help="face recognition")
ap.add_argument("-t","--folder", required=True,help="path to data")
args = vars(ap.parse_args())


print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()                            	# initialize dlib's face detector (HOG-based) and then create

predictor = dlib.shape_predictor(args["shape_predictor"])                    	# the facial landmark predictor

facerec = dlib.face_recognition_model_v1(args["fac"])                      #the resnet model for face descriptor generatiom


print("[INFO] camera sensor warming up...")								

cap = cv2.VideoCapture(1)											# initialize the video stream 

time.sleep(2.0)														#allow the cammera sensor to warmup	by putting a delay of 2 milliseconds				
	

m = 0
c = 0
temp = []



while(m == 0)  :                   # loop over the frames from the video stream as long as the variable m = 0



	ret , frame = cap.read()              #capture the frame from the camera into a variable matrix "frame"

	frame = imutils.resize(frame, width=400)      #resizing to a width of 400pixels

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)           #converting the image to grayscale
 

	rects = detector(gray, 1)                       			# detect faces in the grayscale frame

	print("number of faces: {}".format(len(rects)))               #print the number of faces


	for rect in rects:     			# loop over the face detections


		shape = predictor(gray, rect)                   #the output of predictor("","") is in the form of a dlib vector which has the (x, y) coordinates of the 64 landmarks 

		shape1 = face_utils.shape_to_np(shape)          #converting the dlib vector into a numpy array for the purpose of operations
 	
		face_descriptor = facerec.compute_face_descriptor(frame,shape)   #creating a unique vector for the landmarks having 128 elements
	

		array = np.zeros((128,1), dtype=float)       #declaring a numpy array

		for j in range(0,128):                       #for converting the above dlib vector into a numpy array for further operations
			array[j] = face_descriptor[j]



		i = 0
		cost = np.ones((1 , 1000) , dtype = float)      #declaring a numpy array of 1s 


		alist = []                                      #declaring a numpy list

		for q in glob.glob(os.path.join(args["folder"], "*.txt")):         #iterating through the folder containg the data set of customers
			with open(q , 'rb') as fp:
				ar= np.loadtxt(fp , dtype = float)
			fp.close()

			a = dist.euclidean(array , ar) 									#calculating the euclidean distance between the newly generated vector and the vector contained in the current data file
			alist.append(q)													#appending the current file name in the list						
			cost[0][i] = a      											#saving the distance in the numpy array cost created earlier
			i = i + 1

		print(cost.min())													#print the minimum distance									

		print(cost.argmin())												#print the position of the minimum didtance in the array

		v = cost.argmin()
		
		str1 = "data"
		str2 = os.path.splitext(alist[v])[0]							#removing the extension from the file name to get the name of the person										

		if cost[0][v] < 0.50:											#the 0.5 in th statement is the threshold set for an ideal situation , it can be changed as per the convinience
			print str2.replace("data/" , "")							#removing the string "data" from the file name to get the name of the person's name

			file = open("result.txt" , "a")								#print the result from every frame into a file result.txt
			file.write(alist[v])

			fil = open("list.txt" , "a")								#print the final list of persons into a file list.txt
			temp.append(alist[v])

			if(c == 0):
				fil.write(temp[c])

			if(c != 0):

				if (temp[c] != temp[c-1]):
					fil.write(temp[c])
			

			c = c + 1	


			
		else:
			print("you are unknown")

		cv2.imshow("Frame", frame)												#display the current frame
	
		key = cv2.waitKey(1)													#close the window and exit the loop on pressing the key "q"
		if key == ord("q"):
			cv2.destroyAllWindows()								
			m = 1



cap.release()																#release the camera cap