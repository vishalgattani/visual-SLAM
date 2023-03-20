from build.g2opy import g2o
import sys
from utils_sys import Printer
if sys.version_info[0] != 3:
    Printer.red("This script requires Python 3.")
    exit()
else:
    Printer.green("Python 3 exists.")

import numpy as np
import cv2

from feature_extractor import Frame, denormalize, match_frames, IRt

# camera intrinsics
W = 1241//2
H = 376//2
fx = 718.856
fy = 718.856
cx = 607.1928
cy = 185.2157

cam = np.array([[fx,0,W//2],[0,fy,H//2],[0,0,1]])

frames = []
def process_frame(img):
    img = cv2.resize(img,(W,H))
    frame = Frame(img,cam)
    frames.append(frame)
    if len(frames)<=1:
        frame.pose = IRt
        return
    # find the keypoints and descriptors with ORB

    pts,Rt = match_frames(frames[-1],frames[-2])
    # 3d point cloud - triangulate

    pts4d = cv2.triangulatePoints(IRt[:3],Rt[:3],pts[:,0].T,pts[:,1].T).T
    # Printer.orange(Rt.shape)
    frames[-1].pose = np.dot(Rt,frames[-2].pose)

    #reject points
    rempts = np.abs(pts4d[:,3])>0.005
    pts4d = pts4d[rempts]
    # homogenous 3d
    pts4d /= pts4d[:,3:]
    # remove points behind cam if any
    frontpts4d = pts4d[:,2]>0
    pts4d = pts4d[frontpts4d]

    for pt1,pt2 in pts:
        u1,v1 = denormalize(cam,pt1)
        u2,v2 = denormalize(cam,pt2)
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






