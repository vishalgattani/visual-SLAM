# visual-SLAM
Visual SLAM


The fundamental matrix is a 3x3 matrix that relates corresponding image points in two views taken by a calibrated or uncalibrated camera. It can be computed using the 8-point algorithm or other methods. The fundamental matrix encodes the epipolar geometry of the two views, meaning that it determines the relationship between the two views' image planes and the corresponding 3D world points.

The essential matrix is a 3x3 matrix that relates the observations in two views of a scene when the cameras are calibrated. It can be computed from the fundamental matrix and the camera matrices, and it encodes the essential information about the 3D structure of the scene, including the relative scale of the scene, the orientation of the two cameras, and the position of the scene relative to the cameras. The essential matrix is used to extract the relative motion and the 3D structure of the scene, which can then be used to track the camera motion and build a 3D map of the environment in visual slam.