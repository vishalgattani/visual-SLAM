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

class Camera:
    def __init__(self, width, height, focal,
                 cx, cy, k1 = 0.0, k2 = 0.0, p1 = 0.0, p2 = 0.0, k3 = 0.0):
        self.width = width
        self.height = height
        self.focal = focal
        self.cx = cx
        self.cy = cy
        self.d = [k1, k2, p1, p2, k3]
        self.matrix = np.array([[fx,0,W//2],[0,fy,H//2],[cx,cy,1]])

class Frame(object):
    def __init__(self,mapp,img) -> None:
        self.kps,self.des = self.extract(img)
        self.id = len(mapp.frames)
        self.orb = cv2.ORB_create(1000)
        mapp.append_frames(self)

    def extract(self,img):
        orb = cv2.ORB_create(1000)
        # detection
        pts = cv2.goodFeaturesToTrack(np.mean(img,axis=2).astype(np.uint8),3000,qualityLevel=0.01,minDistance=3)
        # extraction
        kps = [cv2.KeyPoint(x=f[0][0],y=f[0][1],size=20) for f in pts]
        # orb
        kps,des = orb.compute(img,kps)
        kps = np.array([(kp.pt[0],kp.pt[1]) for kp in kps])
        return kps,des

class Map(object):
    def __init__(self):
        self.frames = []
        self.points = []

    def append_frames(self,frame):
        self.frames.append(frame)


def process_frame(mapp,cam,img):
    matches = fe.extract(img)

    Printer.cyan("%d matches" % (len(matches)))

    for pt1, pt2 in matches:
        u1,v1 = map(lambda x: int(round(x)), pt1)
        u2,v2 = map(lambda x: int(round(x)), pt2)
        cv2.circle(img, (u1, v1), color=(0,255,0), radius=3)
        cv2.line(img, (u1, v1), (u2, v2), color=(255,0,0))

    cv2.imshow("visual-SLAM",img)

cam = Camera(W,H,fx,cx,cy)
mapp = Map()
fe = Extractor()


if __name__ == "__main__":
    Printer.green("visual-SLAM")
    cap = cv2.VideoCapture("./video/kitti_00.mp4")
    while cap.isOpened():
        ret,frame = cap.read()
        if ret:
            process_frame(mapp,cam,frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
