#!/usr/bin/env python
# coding=utf-8
import numpy as np
import cv2
import math
import sys
import xml.etree.ElementTree as ET
import xml.dom.minidom
from xml.dom import minidom
import sys

video = cv2.VideoCapture(-1)
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

def px_distance(distance, picture):
	global click
	if(click==2):
		click = 0
		distance = np.sqrt((px[-1][-1]-px[-2][-1])**2 + ((px[-1][-2]-px[-2][-2])**2)) # sqrt(dy²+dx²)
		print "\nPixel distance:", distance, "\n"
	if(distance!=0 and click == 0):
		cv2.line(picture,(px[-1][-2],px[-1][-1]),(px[-2][-2],px[-2][-1]),(0,0,255),5)
	return distance

def draw_line():
	global video, pic,ret
	distance = 0
	key = 0
	cv2.namedWindow('Choose a color by a double click')
	cv2.setMouseCallback('Choose a color by a double click',coordinates)
	switch = '0:Colored \n1:GrayScale'
	cv2.createTrackbar(switch, 'Choose a color by a double click',0,1,f)
	while(key != 27):
		ret, frame = video.read()
		pic = cv2.flip(frame,1)
		trackbar = cv2.getTrackbarPos(switch,'Choose a color by a double click')
		if(trackbar!=0):
			pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
		distance = px_distance(distance, pic)
		cv2.imshow('Choose a color by a double click', pic)
		key = cv2.waitKey(20)
	video.release()
	cv2.destroyAllWindows()

def calibration(i): #repeat 5 times
	global pic, ret, video
	num_img = 0
	key = 0
	distance = 0
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	# do not forget to change chess dimensions
	objp, objpoints,imgpoints = obj_parameters()
	video = cv2.VideoCapture(0)
	cv2.namedWindow('Raw')
	cv2.setMouseCallback('Raw',coordinates)
	print('Press ESC to stop the calibration\n\n\n')
	print 'Press "r" to record after the detection marker apears.\n','OBS: Please rec at least 15 times.'
	while(key!=27):
		(ret, img) = video.read()
		pic = cv2.flip(img,1)
		gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
		ret, corners = cv2.findChessboardCorners(gray, (8,6), None)#(8,6) Obj square points
		distance = px_distance(distance,pic)
		if ret == True:
			if(cv2.waitKey(20) == 114):
				objpoints.append(objp) # Square coordinates, 0 to 48 in (6,8) matrix
				imgpoints.append(corners) # Pixel values according to objpoints
				num_img += 1
			corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
			cv2.drawChessboardCorners(pic, (8,6), corners2, ret)
		cv2.imshow('Raw', pic)
		key = cv2.waitKey(20)
	video.release()
	cv2.destroyAllWindows()
	#### PARTE 3
	# Find the rotation and translation vectors.
		#ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
	if(num_img > 0):
		print num_img,"\n"
		print "Wait a second..."
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
		# mtx = K matrix (Intrinsic Parameters), rvecs = R and tvecs = t (External Parameters)
		h, w = img.shape[:2]
		newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
		## Python dictionary data structure:
		Camera = {"Original Matrix" : mtx, "Distortion" : dist, "Radial Distortion":rvecs, "Tangencial Distortion": tvecs, "Improved Matrix": newcameramtx, "Region of Interest": roi}
		save_xml(ret,mtx,dist,rvecs,tvecs,newcameramtx,roi,i)
		errors(objpoints,imgpoints,Camera)
		return img, Camera

def obj_parameters():
	objp = np.zeros((8*6,3), np.float32)
	objp[:,:2] = np.mgrid[0:6,0:8].T.reshape(-1,2)
	objp = objp.reshape(-1,1,3)
	objpoints = [] # 3d point in real world space
	imgpoints = [] # 2d points in image plane.
	return objp,objpoints,imgpoints

def errors(objpoints,imgpoints,Camera):
	print '\n','Errors:','\n'
	mean_error = 0
	for i in xrange(len(objpoints)):
		imgpoints2, _ = cv2.projectPoints(objpoints[i], Camera['Radial Distortion'][i], Camera['Tangencial Distortion'][i], Camera['Original Matrix'], Camera['Distortion'])
		error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
		mean_error += error
		print( "total error: {}".format(mean_error/len(objpoints)) )

def save_xml(status, mtx,distortion,radial_distortion,tangencial_distortion,mtx_optimized,roi,i):
	name = "parameters{}.xml".format(i)
	data = ET.Element('Camera Parameters')
	items = ET.SubElement(data, 'Data')
	item1 = ET.SubElement(items, 'Status')
	item2 = ET.SubElement(items, 'Original Matrix')
	item3 = ET.SubElement(items, 'Optimized Matrix')
	item4 = ET.SubElement(items, 'Distortion')
	item5 = ET.SubElement(items, 'Radial Distortion')
	item6 = ET.SubElement(items, 'Tangencial Distortion')
	item7 = ET.SubElement(items, 'Region of Interest')
	item1.text = str(status)
	item2.text = str(mtx)
	item3.text = str(mtx_optimized)
	item4.text = str(distortion)
	item5.text = str(radial_distortion)
	item6.text = str(tangencial_distortion)
	item7.text = str(roi)
	mydata = ET.tostring(data, pretty_print=True)
	myfile = open(name, "w")
	myfile.write(mydata)

def obj_measurement(Camera):
	global pic
	square_lenght = 27.7 # In mm
	key = 0
	distance_raw = 0
	distance_undistort = 0
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	objp, objpoints,imgpoints = obj_parameters()
	video = cv2.VideoCapture(0) # takes the video adress
	print('Press ESC to stop the object measurement\n\n\n')
	cv2.namedWindow('Undistort')
	cv2.namedWindow('Raw')
	cv2.setMouseCallback('Raw',coordinates)
	cv2.setMouseCallback('Undistort',coordinates)
	while(key!=27):
		(ret, img) = video.read()
		img = cv2.flip(img,1)
		(ret, pic) = video.read()
		pic = cv2.flip(pic,1)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		ret, corners = cv2.findChessboardCorners(gray, (8,6), None)#(7,6)
		if(ret == True):
			corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
			cv2.drawChessboardCorners(img, (8,6), corners2, ret)
		#z = Z_measurement()
		dst = cv2.undistort(img, Camera['Original Matrix'], Camera['Distortion'], None, Camera['Improved Matrix'])
		distance_raw = px_distance(distance_raw, pic)
		distance_undistort = px_distance(distance_undistort, dst)
		cv2.imshow('Undistort', dst)
		cv2.imshow('Raw',pic)
		key = cv2.waitKey(20)
	video.release()
	cv2.destroyAllWindows()

def load_xml_camera_paremeters():
	ch = []
	tree = ET.parse('parameters0.xml')
	root = tree.getroot()
	for elem in root:
		for subelem in elem:
			print(subelem.text)
			ch.append(float(subelem.text))
	return Camera['Region of Interest'], Camera

def data_analysis(data):
	pass
def main():
	draw_line()
	key = raw_input("\nPress 'C' or 'c' if you want to calibrate the camera: \nOtherwise the camera parameters will be loaded.\n\n")
	if(key == 'C' or key == 'c'):
		data = []
		for i in xrange(5):
			print('Click on the image Box. \n')
			(img,Camera) = calibration(i)
			data.append(Camera)
		data_analysis(data)
	else:
		(img,Camera) = load_xml_camera_paremeters()
	obj_measurement(Camera)
	video.release()

main()
