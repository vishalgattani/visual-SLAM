import cv2
import numpy as np

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, EssentialMatrixTransform


# turn [[x,y]] -> [[x,y,1]]
def add_ones(x):
  return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

class Extractor(object):
    def __init__(self, mapp,cam):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None
        self.K = cam.cam_matrix
        self.Kinv = np.linalg.inv(self.K)
        self.prev_frame = None
        self.current_frame = None
        self.prev_keypoints = None
        self.current_keypoints = None
        self.prev_des = None
        self.current_des = None
        self.matches = None
        self.detector = cv2.GFTTDetector_create(maxCorners=1000, minDistance=15.0,
                                                qualityLevel=0.001, useHarrisDetector=False)
        self.descriptor = cv2.xfeatures2d.BriefDescriptorExtractor_create(bytes=32, use_orientation=False)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.focal = cam.focal
        self.projection_center = (cam.cx, cam.cy)

    def get_keypoints(self):
        kps = self.detector.detect(self.current_frame)
        kps, des = self.descriptor.compute(self.current_frame, kps)

        orb = cv2.ORB_create()
        pts = cv2.goodFeaturesToTrack(np.mean(self.current_frame,axis=2).astype(np.uint8),3000,qualityLevel=0.01,minDistance=3)
        kps = [cv2.KeyPoint(x=f[0][0],y=f[0][1],size=20) for f in pts]
        kps,des = orb.compute(self.current_frame, kps)
        return np.array([(kp.pt[0],kp.pt[1]) for kp in kps]),des

        # return kps, des

    def find_matches(self):
        kp1 = []
        kp2 = []
        self.matches = self.matcher.knnMatch(self.prev_des, self.current_des,k=2)
        # self.matches = sorted(self.matches, key = lambda x:x.distance)
        # for match in self.matches:
        #     kp1.append([self.prev_keypoints[match.queryIdx].pt[0], self.prev_keypoints[match.queryIdx].pt[1]])
        #     kp2.append([self.current_keypoints[match.trainIdx].pt[0], self.current_keypoints[match.trainIdx].pt[1]])
        # return np.array(kp1), np.array(kp2)
        ret = []
        idx1,idx2 = [],[]
        for m,n in self.matches:
        # ratio test as per Lowe's paper
            if m.distance < 0.75*n.distance:
                p1 = self.prev_keypoints[m.queryIdx]
                p2 = self.current_keypoints[m.trainIdx]
                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)
                ret.append((p1,p2))

        ret = np.array(ret)
        idx1 = np.array(idx1)
        idx2 = np.array(idx2)

        model, inliers = ransac((ret[:,0],ret[:,1]),
                        # FundamentalMatrixTransform,
                        EssentialMatrixTransform,
                        min_samples=8,
                        residual_threshold=0.25,
                        max_trials=1000)
        ret = ret[inliers]
        #extract rotational and translational matrices
        EssentialMat = model.params
        return idx1[inliers],idx2[inliers],EssentialMat

    def process_frame(self, mapp,frame):
        mapp.append_frames(frame)
        self.current_frame = frame
        if len(mapp.frames)<=1:
            self.current_keypoints, self.current_des = self.get_keypoints()
        else:
            self.current_keypoints, self.current_des = self.get_keypoints()
            # kp1, kp2, E = self.find_matches()

            # _, rot, trans, mask = cv2.recoverPose(E, kp2, kp1, focal = self.focal, pp = self.projection_center)
            # scale = self.get_absolute_scale(frame_id)
            # self.R = rot.dot(self.R)
            # self.t = self.t + scale * self.R.dot(trans)

        self.prev_frame = self.current_frame
        self.prev_keypoints = self.current_keypoints
        self.prev_des = self.current_des




