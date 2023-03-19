

import sys
from utils_sys import Printer
if sys.version_info[0] != 3:
    Printer.orange("This script requires Python 3.")
    exit()
else:
    Printer.orange("Python 3 exists.")

import numpy as np
import cv2

from feature_extractor import FeatureExtractor

W = 1241//2
H = 376//2
fx = 718.856
fy = 718.856
cx = 607.1928
cy = 185.2157

cam = np.array([[fx,0,W//2],[0,fy,H//2],[0,0,1]])


fe = FeatureExtractor(cam)


def process_frame(img):
    img = cv2.resize(img,(W,H))
    # find the keypoints and descriptors with ORB
    matches,Rt = fe.extract(img)
    print(len(matches),"matches")

    for p1,p2 in matches:
        u1,v1 = fe.denormalize(p1)
        u2,v2 = fe.denormalize(p2)
        cv2.circle(img,(u1,v1),color=(0,255,0),radius=3)
        cv2.line(img,(u1,v1),(u2,v2),color=(255,0,0))
    cv2.imshow("visual-SLAM",img)


if __name__ == "__main__":

    # record video input
    cap = cv2.VideoCapture("./video/kitti_00.mp4")
    while cap.isOpened():
        ret,frame = cap.read()
        if ret:
            process_frame(frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break






