import numpy as np
import cv2
from skimage.measure import ransac
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

class FeatureExtractor(object):
    def __init__(self) -> None:
        self.orb = cv2.ORB_create(100)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None

    def extract(self,img):
        # detection
        features = cv2.goodFeaturesToTrack(np.mean(img,axis=2).astype(np.uint8),3000,qualityLevel=0.01,minDistance=3)
        # extraction
        kps = [cv2.KeyPoint(x=f[0][0],y=f[0][1],size=20) for f in features]
        kps,des = self.orb.compute(img,kps)
        # matching
        ret = []
        if self.last is not None:
            matches = self.bf.knnMatch(des,self.last['des'],k=2)
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    kp1 = kps[m.queryIdx].pt
                    kp2 = self.last["kps"][m.trainIdx].pt
                    ret.append((kp1,kp2))

        ret = np.array(ret)

        #filter
        # how ponts correspond to each other is governed by fundamental matrix
        # Estimate the epipolar geometry between the left and right image.
        if len(ret)>0:
            model, inliers = ransac((ret[:,0],ret[:,1]),
                            FundamentalMatrixTransform, min_samples=8,
                            residual_threshold=1, max_trials=1000)
            ret = ret[inliers]
        self.last = {"kps":kps,"des":des}
        return ret
