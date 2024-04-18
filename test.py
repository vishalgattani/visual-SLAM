from build.g2opy import g2o
import pypangolin as pangolin
import cv2
from utils_sys import Printer
import numpy as np
from extractor import Extractor

from skimage.measure import ransac
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform

# camera intrinsics
W = 1241.0
H = 376.0
fx = 718.856
fy = 718.856
cx = 607.1928
cy = 185.2157
cx = 0
cy = 0

class Camera:
    def __init__(self, width, height, focal,
                 cx, cy, k1 = 0.0, k2 = 0.0, p1 = 0.0, p2 = 0.0, k3 = 0.0):
        self.width = width
        self.height = height
        self.focal = focal
        self.cx = cx
        self.cy = cy
        self.d = [k1, k2, p1, p2, k3]
        self.cam_matrix = np.array([[fx,0,W//2],[0,fy,H//2],[cx,cy,1]])


class Map(object):
    def __init__(self):
        self.frames = []
        self.points = []

    def append_frames(self,frame):
        self.frames.append(frame)



cam = Camera(W,H,fx,cx,cy)
mapp = Map()
fe = Extractor(mapp,cam)


if __name__ == "__main__":
    Printer.green("visual-SLAM")
    cap = cv2.VideoCapture("./video/kitti_00.mp4")
    while cap.isOpened():
        ret,frame = cap.read()
        if ret:
            fe.process_frame(mapp,frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
