import numpy as np
import cv2
from utils_sys import Printer
from skimage.measure import ransac
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform


IRt = np.eye(4)


def add_ones(x):
  if len(x.shape) == 1:
    return np.concatenate([x,np.array([1.0])], axis=0)
  else:
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)


def extractRt(E):
    #computing cameras from E, finding multiple solutions of rotational and translational matrix?
    # Printer.green(E)
    U,D,Vt = np.linalg.svd(E)
    assert np.linalg.det(U)>0
    if np.linalg.det(Vt) < 0:
        Vt *= -1.0
    #Find R and T from Hartley & Zisserman
    W=np.mat([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
    R = np.dot(np.dot(U,W),Vt)
    if np.sum(R.diagonal()<0):
        R = np.dot(np.dot(U,W.T),Vt)
    t = U[:,2] #u3 normalized.
    ret = np.eye(4)
    ret[:3,:3]= R
    ret[:3,3]= t

    # Rt = np.concatenate([R,t.reshape(3,1)],axis=1)
    return ret


def extract(img):
    orb = cv2.ORB_create(100)
    # detection
    pts = cv2.goodFeaturesToTrack(np.mean(img,axis=2).astype(np.uint8),3000,qualityLevel=0.01,minDistance=3)
    # extraction
    kps = [cv2.KeyPoint(x=f[0][0],y=f[0][1],size=20) for f in pts]
    # orb
    kps,des = orb.compute(img,kps)
    return np.array([(kp.pt[0],kp.pt[1]) for kp in kps]),des


def normalize(Kinv,pts):
    return np.dot(Kinv,add_ones(pts).T).T[:,0:2]

def denormalize(K,pt):
    ret = np.dot(K,np.array([pt[0],pt[1],1.0]))
    ret /= ret[2]
    return int(round(ret[0])),int(round(ret[1]))

def match_frames(f1,f2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    # matching
    ret = []
    matches = bf.knnMatch(f1.des,f2.des,k=2)
    for m,n in matches:
        # ratio test as per Lowe's paper
        if m.distance < 0.75*n.distance:
            p1 = f1.pts[m.queryIdx]
            p2 = f2.pts[m.trainIdx]
            ret.append((p1,p2))

    assert len(ret)>=8
    ret = np.array(ret)

    #filter
    # how ponts correspond to each other is governed by fundamental matrix
    # Estimate the epipolar geometry between the left and right image.
    Rt = None
    model, inliers = ransac((ret[:,0],ret[:,1]),
                    # FundamentalMatrixTransform,
                    EssentialMatrixTransform,
                    min_samples=8,
                    residual_threshold=0.0015,
                    max_trials=1000)
    ret = ret[inliers]
    #extract rotational and translational matrices
    Rt = extractRt(model.params)
    return ret,Rt

class Frame(object):
    def __init__(self,img,K) -> None:
        self.K = K
        self.pose = IRt
        self.Kinv = np.linalg.inv(self.K)
        pts,self.des = extract(img)
        self.pts = normalize(self.Kinv,pts)

