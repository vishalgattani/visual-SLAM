from build.g2opy import g2o

import pypangolin as pango
from OpenGL.GL import *
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

class Map():
    def __init__(self) -> None:
        self.frames = []
        self.points = []

    def display(self):
        for f in self.frames:
            print(f.id,f.pose)

mapp = Map()

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
    def __init__(self,mapp,loc) -> None:
        self.frames = []
        self.xyz = loc
        self.idxs = []
        self.id = len(mapp.points)
        mapp.points.append(self)

    def addobs(self,frame,idx):
        self.frames.append(frame)
        # which des in frame index
        self.idxs.append(idx)



def process_frame(img):
    img = cv2.resize(img,(W,H))
    frame = Frame(mapp,img,cam)
    if frame.id<=1:
        return
    # find the keypoints and descriptors with ORB
    f1 = mapp.frames[-1]
    f2 = mapp.frames[-2]
    idx1,idx2,Rt = match_frames(f1,f2)
    # 3d point cloud - triangulate


    # Printer.orange(Rt.shape)
    f1.pose = np.dot(Rt,f2.pose)
    pts4d = triangulate(f1.pose,f2.pose,f1.pts[idx1],f2.pts[idx2])
    pts4d /= pts4d[:,3:]
    #reject points
    # remove points behind cam if any
    rempts = (np.abs(pts4d[:,3])>0.005) & (pts4d[:,2]>0)
    # pts4d = pts4d[rempts]

    for i,p in enumerate(pts4d):

        pt = Point(mapp,p)
        pt.addobs(f1,idx1[i])
        pt.addobs(f2,idx2[i])

    for pt1,pt2 in zip(f1.pts[idx1],f2.pts[idx2]):
        u1,v1 = denormalize(cam,pt1)
        u2,v2 = denormalize(cam,pt2)
        cv2.circle(img,(u1,v1),color=(0,255,0),radius=3)
        cv2.line(img,(u1,v1),(u2,v2),color=(255,0,0))
    cv2.imshow("visual-SLAM",img)

    mapp.display()

if __name__ == "__main__":

    # record video input
    cap = cv2.VideoCapture("./video/kitti_00.mp4")
    while cap.isOpened():
        ret,frame = cap.read()
        if ret:
            process_frame(frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break






