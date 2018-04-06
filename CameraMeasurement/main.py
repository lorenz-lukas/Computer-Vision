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
px = []
click = 0
def f(x):
	pass

def coordinates(event,x,y,flags,param):
	global px, click
	if(event == cv2.EVENT_LBUTTONDBLCLK):
		px.append((x,y))
		print('Press ESC to exit!\n')
		print 'Coordinates:',(x,y)
		print 'BGR:',px[-1]
		print('Selecione mais um ponto')
		click = click + 1

def draw_line():
	global video, pic,ret,click
	distance  = 0
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
		if(click==2):
			click = 0
			distance = np.sqrt((px[-1][-1]-px[-2][-1])**2 + ((px[-1][-2]-px[-2][-2])**2)) # sqrt(dy²+dx²)
			print "\nPixel distance:", distance, "\n"
		if(distance!=0):
			cv2.line(pic,(px[-1][-2],px[-1][-1]),(px[-2][-2],px[-2][-1]),(0,0,255),5)
			#font = cv2.FONT_HERSHEY_SIMPLEX
			#cv2.putText(pic,'Distance:',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)
		cv2.imshow('Choose a color by a double click', pic)
		key = cv2.waitKey(20)
	video.release()
	cv2.destroyAllWindows()
def calibration():
	print("oi")

def main():
	draw_line()
	calibration()
	video.release()
main()
