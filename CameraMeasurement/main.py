#!/usr/bin/env python
# coding=utf-8
import numpy as np
import cv2
import math
import sys
from scipy.spatial import distance
video = cv2.VideoCapture(-1)
(ret, pic) = video.read()
np.seterr(over='ignore')
px = []
px_und = []
click_und = 0
click = 0
x_raw_real = [0,0]
y_raw_real = [0,0]
x_undistort_real = [0,0]
y_undistort_real = [0,0]
def f(x):
	pass
def coordinates(event,x,y,flags,param):
	global px, click
	if(event == cv2.EVENT_LBUTTONDBLCLK):
		px.append((x,y))
		print('Press ESC to exit!\n')
		print 'Coordinates:',(x,y)
		print('Selecione mais um ponto')
		click = click + 1
def coordinates_undistort(event,x,y,flags,param):
		global px_und, click_und, undistort
		if(event == cv2.EVENT_LBUTTONDBLCLK):
			px_und.append((x,y))
			print('Press ESC to exit!\n')
			print 'Coordinates in Undistort image:',(x,y)
			print('Selecione mais um ponto')
			click_und = click_und + 1

def px_distance(distance, picture):
	global click,click_und, undistort
	if(click==2):
		click = 0
		distance = np.sqrt((px[-1][-1] - px[-2][-1])**2 + ((px[-1][-2] - px[-2][-2])**2)) # sqrt(dy²+dx²)
		real_distance = np.sqrt((x_raw_real[-1] - x_raw_real[-2])**2 + (y_raw_real[-1] - y_raw_real[-2])**2)
		print "\nPixel distance in raw:", distance, "\n"
		print "\nReal distance:", real_distance, "\n"
	if(click_und == 2):
		click_und = 0
		distance = np.sqrt((px_und[-1][-1] - px_und[-2][-1])**2 + ((px_und[-1][-2]-px_und[-2][-2])**2)) # sqrt(dy²+dx²)
		real_distance = np.sqrt((x_undistort_real[-1] - x_undistort_real[-2])**2 + (y_undistort_real[-1] - y_undistort_real[-2])**2)
		print "\nPixel distance in undistort:", distance, "\n"
		print "\nReal distance in undistort:", real_distance, "\n"
		distance = 0
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

def calibration(j,i): #repeat 5 times
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
	while(num_img < 20):
		(ret, img) = video.read()
		pic = cv2.flip(img,1)
		gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
		ret, corners = cv2.findChessboardCorners(gray, (8,6), None)#(8,6) Obj square points
		distance = px_distance(distance,pic)
		if ret == True:
			objpoints.append(objp) # Square coordinates, 0 to 48 in (6,8) matrix
			imgpoints.append(corners) # Pixel values according to objpoints
			num_img += 1
			corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
			cv2.drawChessboardCorners(pic, (8,6), corners2, ret)
		cv2.imshow('Raw', pic)
		key = cv2.waitKey(20)
	cv2.waitKey(500)
	video.release()
	cv2.destroyAllWindows()
	if(num_img > 0):
		print "Wait a second..."
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
		# mtx = K matrix (Intrinsic Parameters), rvecs = R and tvecs = t (External Parameters)
		h, w = img.shape[:2]
		newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
		## Python dictionary data structure:
		Camera = {"Ret": ret,"Original Matrix" : mtx, "Distortion" : dist, "Rotational Distortion":rvecs, "Translation Distortion": tvecs, "Improved Matrix": newcameramtx, "Region of Interest": roi}
		tvecs,rvecs = Intrinsics(Camera,objp,corners2,Camera['Original Matrix'],Camera['Distortion'])
		save_xml(ret,mtx,dist,rvecs,tvecs,newcameramtx,roi,j,i)
		#errors(objpoints,imgpoints,Camera)
		return img, Camera

def Intrinsics(Camera,objp,corners2,mtx,dist):
	#### PARTE 3
	#rvecsRansac, tvecsRansac, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
	ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
	print "Computed by: calibrateCamera, solvePnp and solvePnPRansac\n"
	print "Ret:\n", Camera['Ret'],
	print "\n Ret:\n", ret
	#Camera['Ret'] = ret
	#print "Ret:\n", inliers
	print "Rotational Distortion:\n", Camera['Rotational Distortion']
	print "Rotational Distortion:\n", rvecs
	Camera['Rotational Distortion'] = rvecs
	#print "Radial Distortion:\n", rvecsRansac
	print "Translation Distortion:\n", Camera['Translation Distortion']
	print "Translation Distortion:\n", tvecs
	#print "Tangencial Distortion:\n", tvecsRansac
	Camera['Translation Distortion'] = tvecs
	return tvecs, rvecs
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
		imgpoints2, _ = cv2.projectPoints(objpoints[i], Camera['Rotational Distortion'][i], Camera['Translation Distortion'][i], Camera['Original Matrix'], Camera['Distortion'])
		error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
		mean_error += error
		print( "total error: {}".format(mean_error/len(objpoints)) )

def save_xml(ret, mtx,distortion,rotational_distortion,translation_distortion,mtx_optimized,roi,j,i):
	index = j*10+i
	fs_write = cv2.FileStorage("parameters{}.xml".format(index), cv2.FILE_STORAGE_WRITE)
	fs_write.write("Ret", ret)
	fs_write.write("Original_Matrix",mtx)
	fs_write.write("Optimized_Matrix",mtx_optimized)
	fs_write.write("Distortion",distortion)
	fs_write.write("Rotation_Distortion",np.array(rotational_distortion))
	fs_write.write("Translation_Distortion",np.array(translation_distortion))
	fs_write.write("ROI", roi)
	fs_write.release()
	fs_write = cv2.FileStorage("intrinsics{}.xml".format(index), cv2.FILE_STORAGE_WRITE)
	fs_write.write("Ret", ret)
	fs_write.write("Original_Matrix",mtx)
	fs_write.write("Optimized_Matrix",mtx_optimized)
	fs_write.write("Distortion",distortion)
	fs_write.release()
	fs_write = cv2.FileStorage("extrinsics{}.xml".format(index), cv2.FILE_STORAGE_WRITE)
	fs_write.write("Rotation_Distortion",np.array(rotational_distortion))
	fs_write.write("Translation_Distortion",np.array(translation_distortion))
	fs_write.write("ROI", roi)
	fs_write.release()

def load_xml_camera_paremeters():
	fs_read = cv2.FileStorage('CameraParameters.xml', cv2.FILE_STORAGE_READ)
	Camera = {"Ret": fs_read.getNode('RET').mat(),"Original Matrix" : fs_read.getNode('Original_Matrix').mat(), "Distortion" :fs_read.getNode('Distortion').mat(), "Rotational Distortion":fs_read.getNode('Rotational_Distortion').mat(), "Translation Distortion":fs_read.getNode('Translation_Distortion').mat(),"Improved Matrix": fs_read.getNode('Improved_Matrix').mat(), "Region of Interest": fs_read.getNode('ROI').mat()}
	fs_read.release()
	return Camera['Original Matrix'], Camera
def measure(distance_raw,distance_undistort):
	global undistort
	if(undistort):
		pass
	else:
		pass
#def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
#	return (knownWidth * focalLength) / perWidth
def projectpoints_undistort(dis,x_img,y_img):
	global x_undistort_real, y_undistort_real
	r = x_img**2+y_img**2
	factor = 1 + dis[0][0]*r**2+dis[0][1]*r**4+dis[0][4]*r**6
	x_undistort_real.append(factor*x_img + 2*dis[0][2]*x_img*y_img + dis[0][3]*(r**2+2*x_img**2))
	y_undistort_real.append(factor*y_img + 2*dis[0][3]*x_img*y_img + dis[0][2]*(r**2+ 2*y_img**2))

def projectpoints_raw(Camera,x_img,y_img):
	global x_raw_real,y_raw_real

def obj_measurement(Camera): # Reference object dimensions: height = 215mm, windth = 88mm
	global pic,px,px_und, click, click_und
	square_lenght = 27.7 # In mm
	key = 0
	flag = 1
	distance_raw = 0
	distance_undistort = 0
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	objp, objpoints,imgpoints = obj_parameters()
	axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
	video = cv2.VideoCapture(0) # takes the video adress
	print('Press ESC to stop the object measurement\n\n\n')
	cv2.namedWindow('Undistort')
	cv2.namedWindow('Raw')
	cv2.setMouseCallback('Raw',coordinates)
	cv2.setMouseCallback('Undistort',coordinates_undistort)
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
			# project 3D points to image plane
			imgpts, jac = cv2.projectPoints(axis, np.array(Camera['Rotational Distortion']), np.array(Camera['Translation Distortion']), np.array(Camera['Original Matrix']), np.array(Camera['Distortion']))
		dst = cv2.undistort(img, Camera['Original Matrix'], Camera['Distortion'], None, Camera['Improved Matrix'])
		if(click_und != 0):
			projectpoints_undistort(Camera['Distortion'],px_und[-1][-2],px_und[-1][-1])
			distance_undistort = px_distance(distance_undistort, dst)
		if(click != 0):
			projectpoints_raw(Camera,px[-1][-2],px[-1][-1])
			distance_raw = px_distance(distance_raw, pic)
		#measure(distance_raw,distance_undistort)
		cv2.imshow('Undistort', dst)
		cv2.imshow('Raw',pic)
		key = cv2.waitKey(20)
	video.release()
	cv2.destroyAllWindows()

def mean(c1,c2,c3,c4,c5,j):
	ret_mean = (np.array(c1['Ret'])+np.array(c2['Ret'])+np.array(c3['Ret'])+np.array(c4['Ret'])+np.array(c5['Ret']))/5
	mtx_mean = (np.array(c1['Original Matrix'])+np.array(c2['Original Matrix'])+np.array(c3['Original Matrix'])+np.array(c4['Original Matrix'])+np.array(c5['Original Matrix']))/5
	optimized_mtx_mean = (np.array(c1['Improved Matrix'])+np.array(c2['Improved Matrix'])+np.array(c3['Improved Matrix'])+np.array(c4['Improved Matrix'])+np.array(c5['Improved Matrix']))/5
	distortion_mean = (np.array(c1['Distortion'])+np.array(c2['Distortion'])+np.array(c3['Distortion'])+np.array(c4['Distortion'])+np.array(c5['Distortion']))/5
	translation_distortion_mean = (np.array(c1['Translation Distortion'])+np.array(c2['Translation Distortion'])+np.array(c3['Translation Distortion'])+np.array(c4['Translation Distortion'])+np.array(c5['Translation Distortion']))/5
	rotational_distortion_mean = (np.array(c1['Rotational Distortion'])+np.array(c2['Rotational Distortion'])+np.array(c3['Rotational Distortion'])+np.array(c4['Rotational Distortion'])+np.array(c5['Rotational Distortion']))/5
	roi_mean = (np.array(c1['Region of Interest'])+np.array(c2['Region of Interest'])+np.array(c3['Region of Interest'])+np.array(c4['Region of Interest'])+np.array(c5['Region of Interest']))/5

	fs_write = cv2.FileStorage("CameraParameters{}.xml".format(j), cv2.FILE_STORAGE_WRITE)
	fs_write.write("Ret", ret_mean)
	fs_write.write("Original_Matrix",mtx_mean)
	fs_write.write("Optimized_Matrix",optimized_mtx_mean)
	fs_write.write("Distortion",distortion_mean)
	fs_write.write("Rotational_Distortion",rotational_distortion_mean)
	fs_write.write("Translation_Distortion",translation_distortion_mean)
	fs_write.write("ROI", roi_mean)
	fs_write.release()

def SD(C1,C2,C3,C4,C5,j):
	pass
def data_analysis(data,j):
	C1 = data[-1]
	C2 = data[-2]
	C3 = data[-3]
	C4 = data[-4]
	C5 = data[-5]
	mean(C1,C2,C3,C4,C5,j)
	#SD(C1,C2,C3,C4,C5,j)
def main():
	draw_line()
	key = raw_input("\nPress 'C' or 'c' if you want to calibrate the camera: \nOtherwise the camera parameters will be loaded.\n\n")
	if(key == 'C' or key == 'c'):
		data = []
		for j in xrange(3):
			print 'Position the calibration reference in 3 seconds:\n'
			for i in xrange(5):
				print('Click on the image Box. \n')
				(img,Camera) = calibration(j,i)
				data.append(Camera)
			data_analysis(data,j)
	else:
		(img,Camera) = load_xml_camera_paremeters()
	obj_measurement(Camera)
	video.release()

main()
