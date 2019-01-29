"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import random
import numpy as np
import matplotlib.pyplot as plt
import math
from helper import _singularize, refineF
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import minimize, least_squares

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation

    pts1_scaled = pts1/M
    pts2_scaled = pts2/M

    A_f = np.zeros((pts1_scaled.shape[0], 9))

    for i in range(pts1_scaled.shape[0]):
        A_f[i, :] = [ pts2_scaled[i,0]*pts1_scaled[i,0] , pts2_scaled[i,0]*pts1_scaled[i,1] , pts2_scaled[i,0], pts2_scaled[i,1]*pts1_scaled[i,0] , pts2_scaled[i,1]*pts1_scaled[i,1] , pts2_scaled[i,1], pts1_scaled[i,0], pts1_scaled[i,1], 1  ]

    # print('A shape: ',A_f.shape)

    u, s, vh = np.linalg.svd(A_f)
    v = vh.T
    f = v[:, -1].reshape(3,3)

    ## NO NEED TO SINGULARIZE, ALREADY BEING SINGULARIZED IN REFINEf
    # f = _singularize(f)
    # print(f)

    # print('rank of f :', np.linalg.matrix_rank(f))

    f = refineF(f, pts1_scaled, pts2_scaled)
    # print('refined f :', f)

    # print('rank of refined f :', np.linalg.matrix_rank(f))

    T =  np.diag([1/M,1/M,1])

    unscaled_F = T.T.dot(f).dot(T)
    # print('unscaled_F :', unscaled_F)

    return unscaled_F

'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):
    # Replace pass by your implementation

    pts1_scaled = pts1 / M
    pts2_scaled = pts2 / M

    A_f = np.zeros((pts1_scaled.shape[0], 9))

    for i in range(pts1_scaled.shape[0]):
        A_f[i, :] = [pts2_scaled[i, 0] * pts1_scaled[i, 0], pts2_scaled[i, 0] * pts1_scaled[i, 1], pts2_scaled[i, 0],
                     pts2_scaled[i, 1] * pts1_scaled[i, 0], pts2_scaled[i, 1] * pts1_scaled[i, 1], pts2_scaled[i, 1],
                     pts1_scaled[i, 0], pts1_scaled[i, 1], 1]

    # print('A: ', A_f)
    # print('A shape: ', A_f.shape)

    u, s, vh = np.linalg.svd(A_f)
    v = vh.T
    f1 = v[:, -1].reshape(3, 3)
    f2 = v[:, -2].reshape(3, 3)

    fun = lambda a: np.linalg.det(a * f1 + (1 - a) * f2)

    a0 = fun(0)
    a1 = (2/3)*( fun(1) - fun(-1))  -  ((fun(2)-fun(-2))/12)
    a2 = 0.5*fun(1) + 0.5*fun(-1) -fun(0)
    a3 = (-1/6)*(fun(1)- fun(-1))  +  (fun(2)-fun(-2))/12

    coeff = [a3, a2, a1, a0]
    # coeff = [a0, a1, a2, a3]   // WRONG
    roots = np.roots(coeff)

    # print('roots: ', roots)

    T = np.diag([1 / M, 1 / M, 1])
    F_list =  np.zeros( (3,3,1) )

    for root in roots:
        if np.isreal(root):
            a = np.real(root)
            F = a*f1 + (1- a)*f2
            # F = refineF(F, pts1_scaled, pts2_scaled)
            unscaled_F = T.T.dot(F).dot(T)
            if np.linalg.matrix_rank(unscaled_F)==3:
                print('---------------------------------------------------------------------------')
                F = refineF(F, pts1_scaled, pts2_scaled)
                unscaled_F = F
            F_list = np.dstack(  (  F_list, unscaled_F)  )

    F_list = F_list[:,:,1:]

    # print('F_list shape: ', F_list.shape)

    return F_list

'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation

    E = K2.T.dot(F).dot(K1)
    return E


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation

    # TRIANGULATION
    # http://cmp.felk.cvut.cz/cmp/courses/TDV/2012W/lectures/tdv-2012-07-anot.pdf

    # Form of Triangulation :
    #
    # x = C.X
    #
    # |x|             | u |
    # |y| =   C(3x4). | v |
    # |1|             | w |
    #                 | 1 |
    #
    # 1 = C_3 . X
    #
    # x_i . (C_3_i.X_i) = C_1_i.X_i
    # y_i.  (C_3_i.X_i) = C_2_i.X_i

    # Subtract RHS from LHS and equate to 0
    # Take X common to get AX=0
    # Solve for X with SVD
    # for 2 points we have four equation

    P_i = []

    for i in range(pts1.shape[0]):
        A = np.array([   pts1[i,0]*C1[2,:] - C1[0,:] ,
                         pts1[i,1]*C1[2,:] - C1[1,:] ,
                         pts2[i,0]*C2[2,:] - C2[0,:] ,
                         pts2[i,1]*C2[2,:] - C2[1,:]   ])

        # print('A shape: ', A.shape)
        u, s, vh = np.linalg.svd(A)
        v = vh.T
        X = v[:,-1]
        # NORMALIZING
        X = X/X[-1]
        # print(X)
        P_i.append(X)

    P_i = np.asarray(P_i)

    # print('P_i: ', P_i)

    # MULTIPLYING TOGETHER WIH ALL ELEMENET OF Ps
    pts1_out = np.matmul(C1, P_i.T )
    pts2_out = np.matmul(C2, P_i.T )

    pts1_out = pts1_out.T
    pts2_out = pts2_out.T

    # NORMALIZING
    for i in range(pts1_out.shape[0]):
        pts1_out[i,:] = pts1_out[i,:] / pts1_out[i, -1]
        pts2_out[i,:] = pts2_out[i,:] / pts2_out[i, -1]

    # NON - HOMOGENIZING
    pts1_out = pts1_out[:, :-1]
    pts2_out = pts2_out[:, :-1]

    # print('pts2_out shape: ', pts2_out.shape)
    # print('pts1_out: ', pts1_out)
    # print('pts2_out: ', pts2_out)

    # CALCULATING REPROJECTION ERROR
    reprojection_err = 0
    for i in range(pts1_out.shape[0]):
        reprojection_err = reprojection_err  + np.linalg.norm( pts1[i,:] - pts1_out[i,:] )**2 + np.linalg.norm( pts2[i,:] - pts2_out[i,:] )**2
    # print(reprojection_err)

    # NON-HOMOGENIZING
    P_i = P_i[:, :-1]

    return P_i, reprojection_err


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''

def makeGaussianFiler(k_size, sigma):
    window = np.zeros( (k_size, k_size) )
    window[k_size//2, k_size//2]=1
    return gaussian_filter( window, sigma)

def epipolarCorrespondence(im1, im2, F, x1, y1):

    # MAKE GAUSSIAN KERNEL
    # kernel_size = 37
    # sigma = 25

    # kernel_size = 39
    # sigma = 17

    # kernel_size = 22
    # sigma = 3

    kernel_size = 51
    sigma = 31

    kernel = makeGaussianFiler(kernel_size, sigma)
    kernel /= np.sum(kernel)
    kernel = np.asarray(kernel)
    kernel = np.dstack( ( kernel, kernel, kernel  )  )
    # print('kernel: ', kernel)
    # plt.imshow(kernel)
    # plt.show()
    # print(kernel.sum())

    # FINDING EPIPOLAR LINE
    sy, sx, _ = im2.shape
    xc = int(x1)
    yc = int(y1)
    v = np.array([xc, yc, 1])
    l = F.dot(v)
    s = np.sqrt(l[0] ** 2 + l[1] ** 2)

    if s == 0:
        error('Zero line vector in displayEpipolar')

    # EQUATION OF LINE IN NORMAL FORM
    l = l / s

    if l[0] != 0:
        ye = sy - 1
        ys = 0
        xe = -(l[1] * ye + l[2]) / l[0]
        xs = -(l[1] * ys + l[2]) / l[0]
    else:
        xe = sx - 1
        xs = 0
        ye = -(l[0] * xe + l[2]) / l[1]
        ys = -(l[0] * xs + l[2]) / l[1]

    N = max( (ye-ys), (xe-xs) )

    x2_list = np.linspace(xs, xe, N)
    y2_list = np.linspace(ys, ye, N)

    x2_list = np.rint(x2_list).astype(int)
    y2_list = np.rint(y2_list).astype(int)

    min_error = np.inf
    x2_min_error = None
    y2_min_error = None

    k_half = kernel_size //2
    k_half__ = (kernel_size-1) // 2
    if x1 >= k_half and y1 >= k_half and x1 <= sx-1-k_half__ and y1 <= sy-1-k_half__:
        patch_1 = im1[y1 - k_half: y1 - k_half + kernel_size, x1 - k_half: x1 - k_half + kernel_size, :]
        patch_1 = np.asarray(patch_1)

        for i in range(x2_list.shape[0]):
            x2 = x2_list[i]
            y2 = y2_list[i]

            if x2 >= k_half and y2 >= k_half and x2 <= sx-1-k_half__ and y2 <= sy-1-k_half__:
                patch_2 = im2[y2-k_half: y2-k_half+kernel_size, x2-k_half: x2-k_half+kernel_size, :]
                patch_2 = np.asarray(patch_2)

                diff = patch_1 - patch_2
                diff_gaussian = np.multiply(kernel, diff)
                err = np.linalg.norm(diff_gaussian)

                if err<min_error:
                    min_error = err
                    x2_min_error = x2
                    y2_min_error = y2

    return x2_min_error, y2_min_error

'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M):
    # Replace pass by your implementation

    max_inliers  =  -np.inf
    inliers_best = np.zeros(pts1.shape[0], dtype=bool)
    points_index_best  = None
    threshold = 1e-3

    epochs = 1000
    for e in range(epochs):
        points_index = random.sample(range(0, pts1.shape[0]), 7)
        # print(points_index)
        sevenpoints_1 = []
        sevenpoints_2 = []
        for point in points_index:
            sevenpoints_1.append(pts1[point, :])
            sevenpoints_2.append(pts2[point, :])
        sevenpoints_1 = np.asarray(sevenpoints_1)
        sevenpoints_2 = np.asarray(sevenpoints_2)

        F_list =  sevenpoint(sevenpoints_1, sevenpoints_2, M)
        for j in range(F_list.shape[2]):
            f = F_list[:, :, j]
            num_inliers = 0
            inliers = np.zeros(pts1.shape[0], dtype=bool)
            for k in range(pts1.shape[0]):
                X2 = np.asarray(  [pts2[k,0], pts2[k,1], 1] )
                X1 = np.asarray(  [pts1[k,0], pts1[k,1], 1] )

                if abs(X2.T.dot(f).dot(X1)) < threshold:
                    num_inliers = num_inliers +1
                    inliers[k] = True
                else:
                    inliers[k] = False

            # print(num_inliers)

            if num_inliers>max_inliers:
                max_inliers = num_inliers
                inliers_best = inliers
                points_index_best = points_index

    print('epoch: ', epochs-1, 'max_inliers: ', max_inliers)
    # print('points_index_best: ', points_index_best)

    # RE-DOING EIGHT POINT ALGO AFTER RANSAC WITH INLIER POINTS
    pts1_inliers= pts1[np.where(inliers_best)]
    pts2_inliers= pts2[np.where(inliers_best)]

    F_best_all_inliers = eightpoint(pts1_inliers, pts2_inliers, M)

    return F_best_all_inliers, inliers_best


def skew(x):
    assert len(x)==3
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation

    theta = np.linalg.norm(r, 2)
    u = r/theta
    u = u.reshape(3,1)
    R = np.eye(3,3)*np.cos(theta) + (1 - np.cos(theta))*(u.dot(u.T)) + skew(u)*(np.sin(theta))
    return R

'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation

    A  =  (R - R.T)/2
    rho = np.asarray([ A[2,1], A[0,2], A[1,0]   ]).T
    s = np.linalg.norm(rho, 2)
    c = ( R[0,0] + R[1,1] + R[2,2] -1)/2
    theta = np.arctan2(s,c)
    u = rho/s

    if s==0 and c==1:
        return np.asarray([0,0,0])

    elif s==0 and c==-1:
        v= (R + np.eye(3)).reshape(9,1)
        u = v/np.linalg.norm(v,2)
        r = u*np.pi
        if np.linalg.norm(r,2)==np.pi and ( (r[0] ==0 and r[1] ==0 and r[2]<0) or ( r[0]==0 and r[1]<0 ) or (r[0]<0) ):
            return -r
        else:
            return r

    elif np.sin(theta) != 0:
        return  u*theta

    else:
        print('No condition satisfied')
        return None


'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation

    # >> > a = np.array([1, 2, 4])
    # >> > a[:, None]  # col
    # array([[1],
    #        [2],
    #        [4]])

    n = p1.shape[0]
    P = np.hstack( ( x[0:n, None], x[n:2*n, None], x[2*n:3*n, None] ) )
    r = x[3*n:3*n+3]
    t = x[3*n+3:3*n+6, None]

    R = rodrigues(r)
    # print('R: ', R)

    M2 =  np.hstack( [R, t] )
    # print(M2)

    C1 = K1.dot(M1)
    C2 = K2.dot(M2)

    # print(P.shape)

    P_homo = np.vstack( [P.T, np.ones(n) ] )
    # print(P_homo.shape)

    p1_hat_homo = np.matmul( C1, P_homo )
    p2_hat_homo = np.matmul( C2, P_homo )

    p1_hat = p1_hat_homo.T
    p2_hat = p2_hat_homo.T

    # NORMALIZING
    for i in range(p1_hat.shape[0]):
        p1_hat[i,:] = p1_hat[i,:] / p1_hat[i, -1]
        p2_hat[i,:] = p2_hat[i,:] / p2_hat[i, -1]

    # NON - HOMOGENIZING
    p1_hat = p1_hat[:, :-1]
    p2_hat = p2_hat[:, :-1]

    residuals = np.concatenate([(p1-p1_hat).reshape([-1]), (p2-p2_hat).reshape([-1])])
    residuals = np.expand_dims(residuals, 1)
    # print('residuals shape: ', residuals.shape, ', n:', n)
    return residuals

'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''

def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation

    R_init = M2_init[:, 0:3]
    r_init = invRodrigues(R_init)
    t_init = M2_init[:, 3].reshape([-1])

    x_init  = np.hstack( [ P_init[:,0].reshape([-1]), P_init[:,1].reshape([-1]), P_init[:,2].reshape([-1]), r_init, t_init ]  )

    # func = lambda x: rodriguesResidual(K1, M1, p1, K2, p2, x)
    # res = least_squares(func, x_init, verbose=2)

    ##### Changing optimizer to minimize
    func = lambda x: (rodriguesResidual(K1, M1, p1, K2, p2, x)** 2).sum()
    print('Running optimizer')
    res = minimize(func, x_init, options={'disp': True})

    x_new = res.x

    print('x_new shape: ', x_new.shape)

    n = p1.shape[0]
    P_new = np.hstack( ( x_new[0:n, None], x_new[n:2*n, None], x_new[2*n:3*n, None] ) )
    r_new = x_new[3*n:3*n+3]
    t_new = x_new[3*n+3:3*n+6, None]

    R_new = rodrigues(r_new)

    M2_new =  np.hstack( [R_new, t_new] )
    print('M2_new: ',M2_new)

    return M2_new, P_new
