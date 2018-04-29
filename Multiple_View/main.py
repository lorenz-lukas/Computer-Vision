#!/usr/bin/env python
# coding=utf-8
import math
import sys
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
pxL = []
pxR = []
sift = cv.xfeatures2d.SIFT_create()
# x = P*X -> pinv(P)*x = X, where P = M*R*t, R = r*alpha.
# x' = P'X
# numpy.linalg.norm((a - b), ord=1)
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

def epipolar(imgL,imgR):
	global sift
	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(imgL,None)
	kp2, des2 = sift.detectAndCompute(imgR,None)
	FLANN_INDEX_KDTREE = 1
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)
	flann = cv.FlannBasedMatcher(index_params,search_params)
	matches = flann.knnMatch(des1,des2,k=2)
	good = []
	pts1 = []
	pts2 = []
	for i,(m,n) in enumerate(matches):
		if m.distance < 0.17*n.distance: # Number of epilines.
			good.append(m)
			pts2.append(kp2[m.trainIdx].pt)
			pts1.append(kp1[m.queryIdx].pt)
	pts1 = np.int32(pts1)
	pts2 = np.int32(pts2)
	F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)
	print F
	# We select only inlier points
	pts1 = pts1[mask.ravel()==1]
	pts2 = pts2[mask.ravel()==1]
	# Find epilines in right image
	lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
	lines1 = lines1.reshape(-1,3)
	img5,img6 = drawlines(imgL,imgR,lines1,pts1,pts2)
	#Find epilines in left image
	lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
	lines2 = lines2.reshape(-1,3)
	img3,img4 = drawlines(imgR,imgL,lines2,pts2,pts1)
	plt.subplot(221),plt.imshow(img5)
	plt.subplot(222),plt.imshow(img3)
	plt.subplot(223),plt.imshow(img6)
	plt.subplot(224),plt.imshow(img4)
	plt.show()

def drawlines(imgL,imgR,lines,pts1,pts2):
    r,c = imgL.shape
    imgL = cv.cvtColor(imgL,cv.COLOR_GRAY2BGR)
    imgR = cv.cvtColor(imgR,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(imgL, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(imgL,tuple(pt1),5,color,-1)
        img2 = cv.circle(imgR,tuple(pt2),5,color,-1)
    return img1,img2

def stereo_map(imgL,imgR):
	stereo = cv.StereoBM_create(numDisparities=128, blockSize=25)
	disparity = stereo.compute(imgL,imgR)
	plt.imshow(disparity,'gray')
	plt.colorbar()
	plt.title('Depth Image')
	plt.show()

def distance_3d(f,b):#f = 25 mm b = 120 mm
    global pxL, pxR
    Z = b*f/(pxL[-1][-2] - pxR[-1][-2])
    X = 0.5*b*((pxL[-1][-2] + pxR[-1][-2])/(pxL[-1][-2] - pxR[-1][-2]))
    Y = 0.5*b(pxL[-1][-1] + pxR[-1][-1]/(pxL[-1][-2] - pxR[-1][-2]))
    return X,Y,Z

def stereo_cameras():
	key = input("Would you like to load images (press l) or take pictures (press any other)?")
	if(key != 'l' or key !='L'):
		print("Press 'q' to quit and 'r' to record.")
		num = 0
		while(num < 2):
			cv2.namedWindow('Press r to record! First the left and then the right image.')
			video = cv2.VideoCapture(-1)
			(ret, pic) = video.read()
			h, w = pic.shape[:2]
			if(key = 'r'):
				if(num == 0):
					cv2.imwrite('left.jpg',pic)
					imgL = pic
					num++
				else
					h, w = pic.shape[:2]
					cv2.imwrite('right.jpg',pic)
					imgR = pic
	else
		imgL = cv2.imread('left.jpg')
		imgR = cv2.imread('right.jpg')
	cv2.destroyAllWindows()
	video.release()
	newcameramtx, roi = cv2.getOptimalNewCameraMatrix(intrinsics, distortion, (w, h), 1, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(intrinsics, distortion, None, newcameramtx, (w, h), 5)
	imgL_undistort = cv2.remap(imgL, mapx, mapy, cv2.INTER_LINEAR)
	imgR_undistort = cv2.remap(imgR, mapx, mapy, cv2.INTER_LINEAR)
	epipolar(imgL_undistort,imgR_undistort)
    stereo_map(imgL_undistort,imgR_undistort)

def load_xml():
	fs = cv2.FileStorage('rotation.xml', cv2.FILE_STORAGE_READ)
    r = fs.getNode('floatdata').mat()
    fs.release()

	fs = cv2.FileStorage('translation.xml', cv2.FILE_STORAGE_READ)
    t = fs.getNode('floatdata').mat()
    fs.release()

	fs = cv2.FileStorage('intrinsics.xml', cv2.FILE_STORAGE_READ)
    intrinsics = fs.getNode('floatdata').mat()
    fs.release()

    fs = cv2.FileStorage('distortion.xml', cv2.FILE_STORAGE_READ)
    distortion = fs.getNode('floatdata').mat()
    fs.release()
	return r,t,intrinsics,distortion

def object_measuremnt_3D():
    cv2.setMouseCallback('Raw',coordinates)
    return X,Y,Z
def main():
    imgL = cv.imread('aloeL.png',0)  #queryimage # left image
    imgR = cv.imread('aloeR.png',0) #trainimage # right image
    epipolar(imgL,imgR)
    stereo_map(imgL,imgR)

	imgL = cv.imread('babyL.png',0)  #queryimage # left image
    imgR = cv.imread('babyR.png',0) #trainimage # right image
    epipolar(imgL,imgR)
    stereo_map(imgL,imgR)

	stereo_cameras()
    #X,Y,Z = object_measuremnt_3D()

main()
