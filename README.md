# 3D-Reconstruction-and-Epipolar-Geometry

In this project we:  

1) Implement the two different methods to estimate the fundamental matrix from corresponding points in two images.   
2) Given the fundamental matrix and calibrated intrinsics, we compute the essential matrix and use this to compute a 3D metric reconstruction from 2D correspondences using triangulation.   
3) We implement a method to automatically match points taking advantage of epipolar constraints and make a 3D visualization of the results.  
4) We implement RANSAC and bundle adjustment to further improve your algorithm.  



Figure 3: Temple images for this assignment
Figure 4: displayEpipolarF in helper.py creates a GUI for visualizing epipolar lines


# 2 Fundamental matrix estimation

In this section you will explore different methods of estimating the fundamental matrix given a pair
of images. In the data/ directory, you will find two images (see Figure 3) from the Middlebury multiview dataset1, which is used to evaluate the performance of modern 3D reconstruction algorithms.

**The Eight Point Algorithm**   
The 8-point algorithm is arguably the simplest method for estimating the fundamental matrix. We use correspondences from data/some corresp.npz.
We implement Function ```F = eightpoint(pts1, pts2, M)```   
where pts1 and pts2 are N × 2 matrices corresponding to the (x; y) coordinates of the N points
in the first and second image repectively. M is a scale parameter.

Some tips:
* We should scale the data, by dividing each coordinate by M (the maximum of the image’s width and height). After computing F, we unscale the fundamental matrix. We enforce the singularity condition of the F before unscaling.   
* We may refine the solution by using local minimization. This probably won’t fix a completely broken solution, but may make a good solution better by locally minimizing a geometric cost function. The helper function refineF in helper.py taking in F and the two sets of points, which we can call from eightpoint before unscaling F.
* Eight-point is just a figurative name, it just means that you need at least 8 points; our algorithm should use an over-determined system (N > 8 points).
* To visualize the correctness of your estimated F, we use the supplied function displayEpipolarF in helper.py, which takes in F, and the two images. This GUI lets you select a point in one of the images and visualize the corresponding epipolar line in the other image.


**The Seven Point Algorithm**   

Since the fundamental matrix only has seven degrees of freedom, it is possible to calculate F using only seven point correspondences. This requires solving a polynomial equation. We Manually select 7 points from  points in
data/some corresp.npz, and use these points to recover a fundamental matrix F. The function
has the signature:
``` Farray = sevenpoint(pts1, pts2, M)```   
where pts1 and pts2 are 7 × 2 matrice s containing the correspondences and M is the normalizer (we use the maximum of the images’ height and width), and Farray is a list array of length either 1 or 3 containing Fundamental matrix/matrices. We use M to normalize the point values between [0; 1] and unnormalize our computed F afterwards.

The algorithm is sensitive to small changes in the point correspondences. We may want to try with different sets of matches.

# 3d Reconstruction

**Metric Reconstruction**
We compute the camera matrices and triangulate the 2D points to obtain the 3D scene structure. To obtain the Euclidean scene structure, we first convert the fundamental matrix F to an essential matrix E. Camera calibration matrices K1 and K2 are known; these are provided in data/intrinsics.npz.

Function to compute the essential matrix E given F, K1 and K2 is written with the signature:
```E = essentialMatrix(F, K1, K2)```    

Given an essential matrix, it is possible to retrieve the projective camera matrices M1 and M2
from it. Assuming M1 is fixed at [I; 0], M2 can be retrieved up to a scale and four-fold rotation
ambiguity. The M1 and M2 here are projection matrices.

Function to triangulate a set of 2D coordinates in the image to a set of 3D points has the signature:
```[P, err] = triangulate(C1, pts1, C2, pts2)```  
where pts1 and pts2 are the N ×2 matrices with the 2D image coordinates and P is an N ×3 matrix
with the corresponding 3D points per row. C1 and C2 are the 3 × 4 camera matrices. We multiply the given intrinsics matrices with your solution for the canonical camera matrices to obtain the final camera matrices

For each point i, we want to solve for 3D coordinates Pi , such that when they are projected back to the two images, they are close to the original 2D points. To project the 3D coordinates back to 2D images, we first write Pi in homogeneous coordinates, and compute C1Pi and C2Pi to obtain the 2D homogeneous coordinates projected to camera 1 and camera 2, respectively.

For each point i, we can write this problem in the following form:
AiPi = 0;   
where Ai is a 4×4 matrix, and Pi is a 4×1 vector of the 3D coordinates in the homogeneous form. Then, we can obtain the homogeneous least-squares solution (discussed in class) to solve for each Pi.


**3D Visualization**
We create a 3D visualization of the temple images. By treating our two images as a stereo-pair, we can triangulate corresponding points in each image, and render their 3D locations.

Function with the signature:
```[x2, y2] = epipolarCorrespondence(im1, im2, F, x1, y1)```  
takes in the x and y coordinates of a pixel on im1 and your fundamental matrix F, and returns the coordinates of the pixel on im2 which correspond to the input point. The match is obtained by computing the similarity of a small window around the (x1; y1) coordinates in im1 to various windows around possible matches in the im2 and returning the closest. Instead of searching for the matching point at every possible location in im2, we can use F and simply search over the set of pixels that lie along the epipolar line (recall that the epipolar line passes through a single point in im2 which corresponds to the point (x1; y1) in im1).
There are various possible ways to compute the window similarity. For this assignment, simple methods such as the Euclidean or Manhattan distances between the intensity of the pixels should suffice. 

Tips:    
* Experiment with various window sizes.
* It may help to use a Gaussian weighting of the window, so that the center has greater influence than the periphery.
* Since the two images only differ by a small amount, it might be beneficial to consider matches for which the distance from (x1; y1) to (x2; y2) is small.

To help test our epipolarCorrespondence,there is a helper function epipolarMatchGUI in python/helper.py, which takes in two images the fundamental matrix. This GUI allows you to click on a point in im1, and will use your function to display the corresponding point in im2. 

It’s not necessary for your matcher to get every possible point right, but it should get easy points (such as those with distinctive, corner-like windows).

These 3D point locations can then plotted using the Matplotlib or plotly package.


**Bundle Adjustment**

In some real world applications, manually determining correspondences is infeasible and often there will be noisy coorespondances. Fortunately, the RANSAC method can be applied to the problem of fundamental matrix estimation.
We Implement the above algorithm with the signature:
```[F, inliers] = ransacF(pts1, pts2, M)```   
where M is defined in the same way as in Section 2 and inliers is a boolean vector of size equivalent to the number of points. Here inliers is set to true only for the points that satisfy the threshold defined for the given fundamental matrix F.

We use the seven point to compute the fundamental matrix from the minimal set of points. Then compute the inliers, and refine your estimate using all the inliers.

So far we have independently solved for camera matrix, Mj and 3D projections, Pi. In bundle adjustment, we will jointly optimize the reprojection error with respect to the points Pi and the camera matrix Mj.
We will be parametrizing the rotation matrix R using Rodrigues formula to produce vector r 2 R3 with funciton:
```R = rodrigues(r)``` as well as the inverse function that converts a rotation matrix R to a Rodrigues vector r with function: ```r = invRodrigues(R)```   

Using this parameterization, we write an optimization function:  
```residuals = rodriguesResidual(K1, M1, p1, K2, p2, x)```    
where x is the flattened concatenation of P, r2, and t2. P are the 3D points; r2 and t2 are the rotation (in the Rodrigues vector form) and translation vectors associated with the projection matrix M2.The residuals are the difference between original image projections and estimated projections (the square of 2-norm of this vector corresponds to the error we computed.

residuals = numpy.concatenate([(p1-p1 hat).reshape([-1]),
(p2-p2 hat).reshape([-1])])
We use this error function and Scipy’s nonlinear least square optimizer leastsq write a function to optimize for the best extrinsic matrix and 3D points using the inlier correspondences from some corresp noisy.npz and the RANSAC estimate of the extrinsics and 3D points as an initialization.   
```[M2, P] = bundleAdjustment(K1, M1, p1, K2, M2 init, p2, P init)```   
