#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 15:42:58 2017
@author: dhingratul

Tracks the WideReciever from a single shot clip, when the offensive team is on
left, using the helper function detect.py

Dependencies
----------
detect.py

Parameters
----------
arg1 : cv2.VideoCapture
    Video File

Returns
-------
void
    Bounding box tracking of wide reciever for the clip

Usage
-------
python tracker_OTS.py -v "Name of video File"
(Example)python tracker_OTS.py -v ./data/clip1.mp4

"""
import cv2
import sys
import argparse
import detect as dt
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=True, help="Name of video clip")
    args = vars(ap.parse_args())
    ctr = 0
    tracker = cv2.Tracker_create("KCF")
    video = cv2.VideoCapture(args["video"])
    print(type(video))
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
    frame_0, frame = video.read()
    if ctr == 0:
        # Define an initial bounding box
        bbox = dt.wideReciever(frame)
        ctr = 1
    if not frame_0:
        print('Error: Unable to read the video file')
        sys.exit()

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
            cv2.rectangle(frame, p1, p2, (0, 0, 255))

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(30) & 0xff
        if k == 27:
                break
