#!/usr/bin/env python
# coding=utf-8

import numpy as np
import cv2
import math
import sys

#variáveis globais devido a função do mouse
video = cv2.VideoCapture(0) # takes the video adress
(ret, pic) = video.read()
np.seterr(over='ignore')
position = 0
#position = []
#position.append(1)
#position.append(1)
flag = 0
def f(x):
	pass

def coordinates(event,x,y,flags,param):
	global flag,position
	(height, width) = pic.shape[:2]

	if(event == cv2.EVENT_LBUTTONDBLCLK):
		px = pic[y,x]
		print('Press ESC to exit!\n')
		print 'Coordinates:',(x,y)
		print 'BGR:',px
		flag = 1
	if(flag):
		px = pic[y,x]
		print(px)
		for i in range(width):
			for j in range(height):
				px2 = pic[j,i]
				if(position == 0):
					distance = np.sqrt(((px[0]-px2[0])**2) + ((px[1]-px2[1])**2)+((px[2]-px2[2])**2))
					if(distance < 13):
						pic[j,i] = [0,0,255]
				else:
					distance = (px-px2)
					if(distance < 13):
						pic[j,i] = 0
		flag = 0
def open():
	global video, pic,ret,position
	#cap = cv2.VideoCapture(0)
	key = 0
	cv2.namedWindow('Choose a color by a double click')
	cv2.setMouseCallback('Choose a color by a double click',coordinates)
	switch = '0:Colored \n1:GrayScale'
	cv2.createTrackbar(switch, 'Choose a color by a double click',0,1,f)

	while(key != 27):
		ret, frame = video.read()
		pic = cv2.flip(frame,1)
		position = cv2.getTrackbarPos(switch,'Choose a color by a double click')
		if(position!=0):
			pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
		cv2.imshow('Choose a color by a double click', pic)
		key = cv2.waitKey(20)
	video.release()
	cv2.destroyAllWindows()

def main():
	open()
	video.release()
main()
