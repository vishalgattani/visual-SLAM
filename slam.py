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

def triangulate(pose1, pose2, pts1, pts2):
  return cv2.triangulatePoints(pose1[:3], pose2[:3], pts1.T, pts2.T).T

class Point(object):
    def __init__(self,loc) -> None:
        self.frames = []
        self.loc = loc
        self.idxs = []

    def addobs(self,frame,idx):
        self.frames.append(frame)
        # which des in frame index
        self.idxs.append(idx)

frames = []
def process_frame(img):
    img = cv2.resize(img,(W,H))
    frame = Frame(img,cam)
    frames.append(frame)
    if len(frames)<=1:
        frame.pose = IRt
        return
    # find the keypoints and descriptors with ORB

    idx1,idx2,Rt = match_frames(frames[-1],frames[-2])
    # 3d point cloud - triangulate


    # Printer.orange(Rt.shape)
    frames[-1].pose = np.dot(Rt,frames[-2].pose)
    pts4d = triangulate(frames[-1].pose,frames[-2].pose,frames[-1].pts[idx1],frames[-2].pts[idx2])
    pts4d /= pts4d[:,3:]
    #reject points
    # remove points behind cam if any
    rempts = (np.abs(pts4d[:,3])>0.005) & (pts4d[:,2]>0)
    # pts4d = pts4d[rempts]

    for i,p in enumerate(pts4d):

        pt = Point(p)
        pt.addobs(frames[-1],idx1[i])
        pt.addobs(frames[-2],idx2[i])

    for pt1,pt2 in zip(frames[-1].pts[idx1],frames[-2].pts[idx2]):
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






