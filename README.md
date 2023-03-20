# visual-SLAM
Visual SLAM (Sparse)

# SLAM Pipeline

1. Feature Extraction - between 2 frames
2.

## Fundamental vs Essential matrix

The fundamental matrix is a 3x3 matrix that relates corresponding image points in two views taken by a calibrated or uncalibrated camera. It can be computed using the 8-point algorithm or other methods. The fundamental matrix encodes the epipolar geometry of the two views, meaning that it determines the relationship between the two views' image planes and the corresponding 3D world points.

In computer vision, the fundamental matrix $\mathbf {F}$  is a 3×3 matrix which relates corresponding points in stereo images. In epipolar geometry, with homogeneous image coordinates, x and x′, of corresponding points in a stereo image pair, Fx describes a line (an epipolar line) on which the corresponding point x′ on the other image must lie. That means, for all pairs of corresponding points holds ${\mathbf  {x}}'^{{\top }}{\mathbf  {Fx}}=0$.

The essential matrix is a 3x3 matrix that relates the observations in two views of a scene when the cameras are calibrated. It can be computed from the fundamental matrix and the camera matrices, and it encodes the essential information about the 3D structure of the scene, including the relative scale of the scene, the orientation of the two cameras, and the position of the scene relative to the cameras. The essential matrix is used to extract the relative motion and the 3D structure of the scene, which can then be used to track the camera motion and build a 3D map of the environment in visual slam.

The essential matrix is a metric object pertaining to calibrated cameras, while the fundamental matrix describes the correspondence in more general and fundamental terms of projective geometry. This is captured mathematically by the relationship between a fundamental matrix \mathbf {F}  and its corresponding essential matrix \mathbf {E} , which is ${\displaystyle \mathbf {E} =({\mathbf {K} '})^{\top }\;\mathbf {F} \;\mathbf {K} }$ where $\mathbf {K}$  and ${\mathbf  {K}}'$ being the intrinsic calibration matrices of the two images involved.

## Estimate rotation and translation using the Essential Matrix

## ORB vs SIFT

SIFT and SURF are patented and you are supposed to pay them for its use. But ORB is not!

ORB is basically a fusion of FAST keypoint detector and BRIEF descriptor with many modifications to enhance the performance. First it use FAST to find keypoints, then apply Harris corner measure to find top N points among them. It also use pyramid to produce multiscale-features. But one problem is that, FAST doesn't compute the orientation.

## Ratio test by Lowe

The ratio test by Lowe is a technique used in feature matching for computer vision tasks such as object recognition and tracking. It is used to find the best match between two sets of features extracted from different images.

The ratio test involves comparing the distance of the closest match to the second closest match for each feature in the first image. If the ratio of the distances is less than a certain threshold value, typically 0.8, then the feature is considered a good match. If the ratio is greater than the threshold, the match is rejected as ambiguous.

The ratio test helps to eliminate false matches and increases the accuracy of feature matching.


## [Camera Calibration and 3D Reconstruction](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html)

The functions in this section use a so-called pinhole camera model. In this model, a scene view is formed by projecting 3D points into the image plane using a perspective transformation.


$$s  \; m' = A [R|t] M'$$


```latex
s  \vecthree{u}{v}{1} = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}
\begin{bmatrix}
r_{11} & r_{12} & r_{13} & t_1  \\
r_{21} & r_{22} & r_{23} & t_2  \\
r_{31} & r_{32} & r_{33} & t_3
\end{bmatrix}
\begin{bmatrix}
X \\
Y \\
Z \\
1
\end{bmatrix}
```

where:
- `(X, Y, Z)` are the coordinates of a 3D point in the world coordinate space
- `(u, v)` are the coordinates of the projection point in pixels
- `A` is a camera matrix, or a matrix of intrinsic parameters
- `(cx, cy)` is a principal point that is usually at the image center
- `fx, fy` are the focal lengths expressed in pixel units.