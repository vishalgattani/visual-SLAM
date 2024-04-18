import sys
from utils_sys import Printer
if sys.version_info[0] != 3:
    Printer.red("This script requires Python 3.")
    exit()
else:
    Printer.green("Python 3 exists.")

import numpy as np
import cv2


def drawOpticalFlowField(img, ref_pts, cur_pts):
    """ Shows a window which shows the optical flow of detected features. """

    # Draw the tracks
    for i, (new, old) in enumerate(zip(cur_pts, ref_pts)):

        a,b = new.ravel()
        c,d = old.ravel()
        v1 = tuple((new - old)*2.5 + old)
        d_v = [new-old][0]*0.75
        arrow_color = (28,24,178)
        arrow_t1 = rotateFunct([d_v], 0.5)
        arrow_t2 = rotateFunct([d_v], -0.5)
        tip1 = tuple(np.float32(np.array([c, d]) + arrow_t1)[0])
        tip2 = tuple(np.float32(np.array([c, d]) + arrow_t2)[0])
        cv2.line(img, v1,(c,d), (0,255,0), 2)
        cv2.line(img, (c,d), tip1, arrow_color, 2)
        cv2.line(img, (c,d), tip2, arrow_color, 2)
        cv2.circle(img, v1,1,(0,255,0),-1)

    cv2.imshow('Optical Flow Field', img)
    cv2.waitKey(1)

    return


if __name__ == "__main__":
    # freeze_support()

    # record video input
    cap = cv2.VideoCapture("./video/kitti_00.mp4")
    while cap.isOpened():
        ret,frame = cap.read()
        if ret:
            cv2.imshow("visual-ODOM",frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
