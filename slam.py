

import sys
if sys.version_info[0] != 3:
    print("This script requires Python 3.")
    exit()
else:
    print("Python 3 exists.")

import numpy as np
import cv2

W = 1920//2
H = 1080//2
# Initiate ORB detector
orb = cv2.ORB_create()


class FeatureExtractor(object):
    def __init__(self) -> None:
        self.orb = cv2.ORB_create(100)
        self.bf = cv2.BFMatcher()
        self.last = None

    def extract(self,img):
        # detection
        features = cv2.goodFeaturesToTrack(np.mean(img,axis=2).astype(np.uint8),3000,qualityLevel=0.01,minDistance=3)
        # extraction
        kps = [cv2.KeyPoint(x=f[0][0],y=f[0][1],size=20) for f in features]
        kps,des = self.orb.compute(img,kps)
        # matching
        matches = None
        if self.last is not None:
            matches = self.bf.match(des,self.last['des'])

        self.last = {"kps":kps,"des":des}
        return kps,des,matches

fe = FeatureExtractor()

def process_frame(img):
    img = cv2.resize(img,(W,H))
    # find the keypoints and descriptors with ORB
    kp, des, matches = fe.extract(img)
    for p in kp:
        u,v = map(lambda u:int(round(u)),p.pt)
        cv2.circle(img,(u,v),color=(0,255,0),radius=3)
    cv2.imshow("visual-SLAM",img)


if __name__ == "__main__":

    # record video input
    cap = cv2.VideoCapture("test_countryroad.mp4")
    while cap.isOpened():
        ret,frame = cap.read()
        if ret:
            process_frame(frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break






