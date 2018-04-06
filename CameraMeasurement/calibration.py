#!/usr/bin/env python
# coding=utf-8
import numpy as np
import cv2 as cv
import glob

num_img = 0
key = 0
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# do not forget to change chess dimensions
objp = np.zeros((8*6,3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:8].T.reshape(-1,2)
objp = objp.reshape(-1,1,3)
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

video = cv.VideoCapture(0) # takes the video adress
cv.namedWindow('Chess')
#images = glob.glob('*.png')
while(key!=27):
    (ret, img) = video.read()
    img = cv.flip(img,1)
    pic = img
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (8,6), None)#(7,6)
    if ret == True:
        #corners2 = cv.cornerSubPix(image=gray, corners=corners, winSize=(11,11), zeroZone = (-1,-1),criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        #imgpoints.append(corners2)
        #objpoints.append(objp[0:corners2.shape[0]])
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        cv.drawChessboardCorners(img, (8,6), corners2, ret)
        num_img += 1
    cv.imshow('Chess', img)
    key = cv.waitKey(20)

video.release()
cv.destroyAllWindows()
if(num_img > 0):
    print num_img,"\n"
    print "Wait a second..."
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None, flags=cv.CALIB_USE_INTRINSIC_GUESS)
    print ret, mtx, dist, rvecs, tvecs
    h, w = pic.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    print "um"

    # ERRO libpng warning: Image width is zero in IHDR
    #libpng warning: Image height is zero in IHDR
    #libpng error: Invalid IHDR data

    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite('calibresult.png', dst)
    print "dois"
    # undistort
    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite('calibresult2.png', dst)
    print "tres"
    mean_error = 0
    for i in xrange(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
        print( "total error: {}".format(mean_error/len(objpoints)) )

#import xml.etree.cElementTree as ET

#root = ET.Element("root")
#doc = ET.SubElement(root, "doc")

#ET.SubElement(doc, "field1", name="blah").text = "some value1"
#ET.SubElement(doc, "field2", name="asdfasd").text = "some vlaue2"

#tree = ET.ElementTree(root)
#tree.write("filename.xml")
