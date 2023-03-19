

import sys
if sys.version_info[0] != 3:
    print("This script requires Python 3.")
    exit()
else:
    print("Python 3 exists.")

import numpy as np
import cv2

from feature_extractor import FeatureExtractor

W = 1920//2
H = 1080//2


fe = FeatureExtractor()

def process_frame(img):
    img = cv2.resize(img,(W,H))
    # find the keypoints and descriptors with ORB
    matches = fe.extract(img)
    print(len(matches),"matches")
    for p1,p2 in matches:
        u1,v1 = map(lambda u:int(round(u)),p1)
        u2,v2 = map(lambda u:int(round(u)),p2)
        cv2.circle(img,(u1,v1),color=(0,255,0),radius=3)
        cv2.line(img,(u1,v1),(u2,v2),color=(255,0,0))
    cv2.imshow("visual-SLAM",img)


if __name__ == "__main__":

    # record video input
    cap = cv2.VideoCapture("test_countryroad.mp4")
    while cap.isOpened():
        ret,frame = cap.read()
        if ret:
            process_frame(frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break






