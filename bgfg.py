#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 10:03:00 2017

@author: dhingratul
Background foreground segmentation using OpenCV example
"""

import cv2
cap = cv2.VideoCapture('clip.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()
while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    cv2.imshow('frame', fgmask)
    # cv2.imwrite("frameDec.jpg",fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
