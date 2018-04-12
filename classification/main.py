#!/usr/bin/env python
# coding=utf-8

import numpy as np
import cv2
import math
import sys

#variáveis globais devido a função do mouse
img = cv2.imread("1.png",cv2.IMREAD_UNCHANGED) # 9
pic = cv2.resize(img, ( 500, 400), interpolation = cv2.INTER_LINEAR)
#pic = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV)

np.seterr(over='ignore')
px = []
classes = []
count = 0
def f(x):
	pass
def coordinates(event,x,y,flags,param):
	global px, pic, classes,count
	if(event == cv2.EVENT_LBUTTONDBLCLK):
		px.append(pic[y][x])
		print('Press ESC to exit!\n')
		print 'Coordinates:',(x,y)
		print 'BGR:',px[-1]
		classes.append(input('diga a classe: 1 = folhagem, 2 = solo, 3 = folha ressecada ou fruto, 4= sombra ou inderterminado '))
		count = count + 1

def main():
	global pic,classes
	key = 0
	cv2.namedWindow('Choose a color by a double click')
	cv2.setMouseCallback('Choose a color by a double click',coordinates)
	while(key != 27 or count == 100):
		#img = cv2.imread("pattern.pdf",cv2.IMREAD_UNCHANGED)
		#pic = cv2.resize(img, ( 500, 400), interpolation = cv2.INTER_LINEAR)
		cv2.imshow('Choose a color by a double click', pic)
		key = cv2.waitKey(20)    # Time im miliseconds where zero means infinite.
	cv2.destroyAllWindows()
	#video.release()

main()
