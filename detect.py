#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 13:00:50 2017

@author: dhingratul
Detects the wide reciever, method adapted from pyimagesearch and OpenCV
documentation
"""

from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
# import argparse

"""
# Argument Parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Name of clip image")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
"""


def wideReciever(image):
    # print("Dimesions of image in Detect {}".format(image.shape))
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    orig = image.copy()
    (rects, weights) = hog.detectMultiScale(image, winStride=(2, 2),
                                            padding=(8, 8), scale=1)

    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # Wide Reciever: The one with the maximum x-cordinate
    idx = np.argmax(pick[:, 1])
    cv2.rectangle(image, (pick[idx, 0], pick[idx, 1]), (pick[idx, 2],
                  pick[idx, 3]), (0, 255, 0), 2)
    bbox = (pick[idx, 0], pick[idx, 1], pick[idx, 2], pick[idx, 3])
     print("BBOX from Detect {}".format(bbox))
    # Display
    # cv2.imshow("Detections", image)
    # cv2.waitKey(0)
    return bbox
"""
# Main
wideReciever(image)
"""
