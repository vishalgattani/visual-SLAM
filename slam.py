from build.g2opy import g2o

import pypangolin as pangolin
import OpenGL.GL as gl
import sys
from utils_sys import Printer
if sys.version_info[0] != 3:
    Printer.red("This script requires Python 3.")
    exit()
else:
    Printer.green("Python 3 exists.")

import numpy as np
import cv2
from multiprocessing import Process,Queue, freeze_support

from feature_extractor import Frame, denormalize, match_frames, IRt

class Map(object):
    def __init__(self):
        self.frames = []
        self.points = []
        # self.q = Queue()
        # self.vp = Process(target=self.viewer_thread,args=(self.q,))
        # self.vp.daemon = True
        # self.vp.start()
        self.viewer_init()
        Printer.green("Starting SLAM Visualizer...")

    def viewer_thread(self,q):
        self.viewer_init()
        while 1 :
            self.viewer_refresh(q)

    def viewer_init(self):
        Printer.green("Starting SLAM Visualizer...")
        pangolin.CreateWindowAndBind('SLAM Visualizer', 640, 480)
        gl.glEnable(gl.GL_DEPTH_TEST)

        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 1000000),
            pangolin.ModelViewLookAt(0, -10, -8,
                               0, 0, 0,
                               0, -1, 0))
        self.handler = pangolin.Handler3D(self.scam)

        # Create Interactive View in window
        self.dcam = (pangolin.CreateDisplay()
                                .SetBounds(
                                    pangolin.Attach(0),
                                    pangolin.Attach(1),
                                    pangolin.Attach(0),
                                    pangolin.Attach(1),
                                    -640.0 / 480.0,
                                )
                                .SetHandler(self.handler))

        self.dcam.Resize(pangolin.Viewport(0,0,640*2,480*2))

    def viewer_refresh(self):
        # while not q.empty():
        #     self.state = q.get()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        self.dcam.Activate(self.scam)

        if self.state is not None:
            if self.state[0].shape[0]>=2:
                ax1 = np.asmatrix(self.state[0][-1:],dtype=np.float32)
                gl.glColor3f(0.0, 1.0, 0.0)
                pangolin.glDrawAxis(ax1,1.0)

            if self.state[0].shape[0]>=1:
                ax1 = np.asmatrix(self.state[0][-1:],dtype=np.float32)
                gl.glColor3f(1.0, 1.0, 0.0)
                pangolin.glDrawAxis(ax1,1.0)

            # if self.state[1].shape[0]!=0:
            #     gl.glPointSize(5)
            #     gl.glColor3f(1.0, 0.0, 0.0)
            #     pangolin.glDrawPoints(self.state[1])
            #     pangolin.glDrawPoints(self.state[2])

        pangolin.FinishFrame()

    def display(self):
        poses, pts = [], []
        for f in self.frames:
            # print(f.id)
            # print(f.pose)
            # print()
            poses.append(f.pose)
        for p in self.points:
            pts.append(p.xyz)
        self.state = np.array(poses), np.array(pts)
        # self.q.put((poses, pts))
        self.viewer_refresh()


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
#   ret = np.zeros((pts1.shape[0], 4))
#   for i, p in enumerate(zip(pts1, pts2)):
#     A = np.zeros((4,4))
#     A[0] = p[0][0] * pose1[2] - pose1[0]
#     A[1] = p[0][1] * pose1[2] - pose1[1]
#     A[2] = p[1][0] * pose2[2] - pose2[0]
#     A[3] = p[1][1] * pose2[2] - pose2[1]
#     _, _, vt = np.linalg.svd(A)
#     ret[i] = vt[3]
#   return ret

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
    if frame.id == 0:
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
    gpts = (np.abs(pts4d[:,3])>0.005) & (pts4d[:,2]>0)
    # pts4d = pts4d[rempts]

    for i,p in enumerate(pts4d):
        if not gpts[i]:
            continue
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
    # freeze_support()

    # record video input
    cap = cv2.VideoCapture("./video/kitti_00.mp4")
    while cap.isOpened():
        ret,frame = cap.read()
        if ret:
            process_frame(frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break






