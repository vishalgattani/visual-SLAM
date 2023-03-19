

import sys
if sys.version_info[0] != 3:
    print("This script requires Python 3.")
    exit()
else:
    print("Python 3 exists.")

import numpy as np
import cv2

def process_frame(img):
    cv2.imshow(img)


if __name__ == "__main__":
    cap = cv2.VideoCapture("test_countryroad.mp4")
    while cap.isOpened():
        ret,frame = cap.read()
        if ret:
            cv2.imshow('visual-SLAM', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break






