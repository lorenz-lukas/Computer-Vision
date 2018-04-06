#!/usr/bin/env python
# Drawing different geometric shapes using OPENCV

import numpy as np
import cv2

# img - image that you want to draw the shapes, color - RGB, thickness - if the argument is '-1'
# the figure will be filled, lineType - (8-connected, anti-aliased - cv2.LINE_AA, ...).

#CREATING A BLOCK IMAGE:
cv2.namedWindow('image')
img = np.zeros((512,512,3), np.uint8)
#DRAWING A DIAGONAL BLUE LINE WITH THICKNESS OF 5px:
cv2.line(img,(0,0),(511,511),(255,0,0),5)
cv2.imshow('image',img)
cv2.waitKey(5000)
cv2.destroyAllWindows()
