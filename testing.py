import cv2
import numpy as np

from functions_lines import findLines

def findlines(bw_image):
	for i in range(50):
		newImage = cv2.GaussianBlur(bw_image,(99,25),0)
		newImage *= 2
	ret, newImage = cv2.threshold(newImage, 80, 255, cv2.THRESH_BINARY)
	cv2.imshow('image', newImage)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	horProj = cv2.reduce(newImage, 1, cv2.REDUCE_AVG)

	# For handwritten text, keep threshold above 0, because, some lines intersect with each other
	# in such a case, 0 may not be seen for the separation between two lines.
	th = 5; # black pixels threshold value. this represents the space lines
	hist = horProj <= th;

	#Get mean coordinate of white white pixels groups
	ycoords = []
	y = 0
	count = 0
	isSpace = False

	for i in range(0, bw_image.shape[0]):
		
		if (not isSpace):
			if (hist[i]): #if space is detected, get the first starting y-coordinates and start count at 1
				isSpace = True
				count = 1
				y = i
		else:
			if (not hist[i]):
				isSpace = False
				#when smoothing, thin letters will breakdown, creating a new blank lines or pixel rows, but the count will be small, so we set a threshold.
				# if (count >= linesThres)
				if (count >= 0):
					ycoords.append(y / count)
			else:
				y = y + i
				count = count + 1

	ycoords.append(y / count)
	
	#returns y-coordinates of the lines found
	return ycoords