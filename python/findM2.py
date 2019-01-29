'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''

import numpy as np
import submission as sub
import helper

def bestM2(pts1, pts2, F, K1, K2):

    # CALCULATE E
    E = sub.essentialMatrix(F, K1, K2)

    # CALCULATE M1 and M2
    M1 = np.array([ [ 1,0,0,0 ],
                    [ 0,1,0,0 ],
                    [ 0,0,1,0 ]  ])

    M2_list = helper.camera2(E)

    #  TRIANGULATION
    C1 = K1.dot(M1)

    P_best = np.zeros( (pts1.shape[0],3) )
    M2_best = np.zeros( (3,4) )
    C2_best = np.zeros( (3,4) )
    err_best = np.inf

    error_list = []

    index = 0
    for i in range(M2_list.shape[2]):
        M2 = M2_list[:, :, i]
        C2 = K2.dot(M2)
        P_i, err = sub.triangulate(C1, pts1, C2, pts2)
        error_list.append(err)
        z_list = P_i[:, 2]
        if all( z>0 for z in z_list):
            index = i
            err_best = err
            P_best = P_i
            M2_best = M2
            C2_best = C2

    # print('error_list: ', error_list)
    # print('err_best: ', err_best)
    # print('M2_best: ', M2_best )
    # print('C2_best: ', C2_best  )
    # print('P_best: ', P_best )
    # print('index: ', index)

    return P_best, C2_best, M2_best, err_best


if __name__ == '__main__':

    data = np.load('../data/some_corresp.npz')

    N = data['pts1'].shape[0]
    M = 640

    pts1 = data['pts1']
    pts2 = data['pts2']

    # EIGHT POINT ALGO
    F = sub.eightpoint(pts1, pts2, M)
    print('F: ', F)

    # GET INTRINSICS
    intrinsics = np.load('../data/intrinsics.npz')
    K1 = intrinsics['K1']
    K2 = intrinsics['K2']

    P_best, C2_best, M2_best, err_best = bestM2(pts1, pts2, F, K1, K2)

    np.savez('../data/q3_3.npz', M2=M2_best, C2=C2_best, P=P_best )
    data = np.load('../data/q3_3.npz')
    print(data.files)




