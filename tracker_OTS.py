#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 15:42:58 2017

@author: dhingratul
Track a player annotated with Bounded box using OpenCV example
"""

import cv2
import sys
 
if __name__ == '__main__' :
     
    tracker = cv2.Tracker_create("KCF")
    video = cv2.VideoCapture("clip.mp4")
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
    frame_init, frame = video.read()
    if not frame_init:
        print('Error: Cannot read video file')
        sys.exit()
     
    # Define an initial bounding box
    #bbox = (287, 23, 86, 320)

    bbox = cv2.selectROI(frame, False)
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
    while True:
        # Read a new frame
        frame_next, frame = video.read()
        if not frame_next:
            break
        # Update tracker
        frame_update, bbox = tracker.update(frame)
 
        # Draw bounding box
        if frame_update:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0,0,255))
 
        # Display result
        cv2.imshow("Tracking", frame)
 
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break
