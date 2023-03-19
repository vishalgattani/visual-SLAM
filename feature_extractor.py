import numpy as np
import cv2
from utils_sys import Printer
from skimage.measure import ransac
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform


def add_ones(x):
   return  np.concatenate([x,np.ones((x.shape[0],1))],axis=1)


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
    Rt = np.concatenate([R,t.reshape(1,3)],axis=0)
    Printer.green(pose)
    return Rt

class FeatureExtractor(object):
    def __init__(self,K) -> None:
        # orb descriptor
        self.orb = cv2.ORB_create(100)
        # sift descriptor
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None
        self.K = K
        self.Kinv = np.linalg.inv(self.K)

    def normalize(self,pts):
        return np.dot(self.Kinv,add_ones(pts).T).T[:,0:2]

    def denormalize(self,pt):
        ret = np.dot(self.K,np.array([pt[0],pt[1],1.0]))
        return int(round(ret[0])),int(round(ret[1]))


    def extract(self,img):

        # detection
        features = cv2.goodFeaturesToTrack(np.mean(img,axis=2).astype(np.uint8),3000,qualityLevel=0.01,minDistance=3)
        # extraction
        kps = [cv2.KeyPoint(x=f[0][0],y=f[0][1],size=20) for f in features]
        # orb
        kps,des = self.orb.compute(img,kps)
        # sift
        #kps,des = self.sift.compute(img,kps)

        # matching
        ret = []
        pts1 = []
        pts2 = []
        if self.last is not None:
            matches = self.bf.knnMatch(des,self.last['des'],k=2)
            for m,n in matches:
                # ratio test as per Lowe's paper
                if m.distance < 0.75*n.distance:
                    kp1 = kps[m.queryIdx].pt
                    kp2 = self.last["kps"][m.trainIdx].pt
                    ret.append((kp1,kp2))



        #filter
        # how ponts correspond to each other is governed by fundamental matrix
        # Estimate the epipolar geometry between the left and right image.
        Rt = None
        if len(ret)>0:
            ret = np.array(ret)

            # normalize coordinates
            ret[:,0,:] = self.normalize(ret[:,0,:])
            ret[:,1,:] = self.normalize(ret[:,1,:])

            model, inliers = ransac((ret[:,0],ret[:,1]),
                            # FundamentalMatrixTransform,
                            EssentialMatrixTransform,
                            min_samples=8,
                            residual_threshold=0.0015,
                            max_trials=1000)
            ret = ret[inliers]
            #extract rotational and translational matrices
            Rt = extractRt(model.params)


        self.last = {"kps":kps,"des":des}
        return ret,Rt
