#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 13:00:50 2017
@author: dhingratul

Helper function to detect the first instance of bounding box for the wide
reciever based on location

Dependencies
----------
imutils

Parameters
----------
arg1 : numpy array
    Image file

Returns
-------
tuple
    Bounding Box detection for wide reciever

Usage
-------
Run from tracker_OTS.py

"""

from imutils.object_detection import non_max_suppression
import numpy as np
import cv2


def wideReciever(image):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    orig = image.copy()
    (tracks, weights) = hog.detectMultiScale(image, winStride=(2, 2),
                                             padding=(8, 8), scale=1)

    for (x, y, w, h) in tracks:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

    tracks = np.array([[x, y, x + w, y + h] for (x, y, w, h) in tracks])
    NMS = non_max_suppression(tracks, probs=None, overlapThresh=0.65)

    # Wide Reciever: The one with the maximum y-cordinate
    idx = np.argmax(NMS[:, 3])
    cv2.rectangle(image, (NMS[idx, 0], NMS[idx, 1]), (NMS[idx, 2],
                  NMS[idx, 3]), (0, 255, 0), 2)
    # Return it as x,y,w,h for Tracker_OTS
    bbox = (NMS[idx, 0], NMS[idx, 1], NMS[idx, 2]-NMS[idx, 0],
            NMS[idx, 3]-NMS[idx, 1])
    return bbox
