

import sys
if sys.version_info[0] != 3:
    print("This script requires Python 3.")
    exit()
else:
    print("Python 3 exists.")

import numpy as np
import cv2

W = 1920//2
H = 1080//2
# Initiate ORB detector
orb = cv2.ORB_create()




def process_frame(img):
    img = cv2.resize(img,(W,H))
    # find the keypoints and descriptors with ORB
    kp, des = orb.detectAndCompute(img,None)
    for p in kp:
        u,v = map(lambda u:int(round(u)),p.pt)
        cv2.circle(img,(u,v),color=(0,255,0),radius=3)
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






