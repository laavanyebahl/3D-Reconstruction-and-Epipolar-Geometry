import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import submission as sub
import helper
import random
import findM2
import visualize
import cv2 as cv2

if __name__ == '__main__':

    data = np.load('../data/some_corresp.npz')
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')

    N = data['pts1'].shape[0]
    M = max(im1.shape[0], im1.shape[1])

    pts1 = data['pts1']
    pts2 = data['pts2']

    intrinsics = np.load('../data/intrinsics.npz')
    K1 = intrinsics['K1']
    K2 = intrinsics['K2']

    # ------------------------------------  (2.1) EIGHT POINT ALGO  ---------------------------
    print('----------------  QUESTION 2.1  ---------------')

    # F = sub.eightpoint(pts1, pts2, M)
    # np.savez('../data/q2_1.npz', F=F, M=M)
    # print(np.load('../data/q2_1.npz').files)
    # print('F from eight point algo: ', F)
    # helper.displayEpipolarF(im1, im2, F)

    # -------------------------------------- (2.2) SEVEN POINT ALGO  -----------------------------
    print('----------------  QUESTION 2.2  ---------------')

    # # points_index = random.sample(range(0, pts1.shape[0]), 7)
    # points_index = [93, 61, 20, 72, 36, 87, 39]
    # print('selected 7 points: ', points_index)
    #
    # sevenpoints_1 = []
    # sevenpoints_2 = []
    #
    # for point in points_index:
    #     sevenpoints_1.append(pts1[point, :])
    #     sevenpoints_2.append(pts2[point, :])
    #
    # sevenpoints_1 = np.asarray(sevenpoints_1)
    # sevenpoints_2 = np.asarray(sevenpoints_2)
    #
    # F_list =   sub.sevenpoint(sevenpoints_1, sevenpoints_2, M)
    # F_sevenpoint = F_list[:,:,0]
    # # F3 gives the best value
    # np.savez('../data/q2_2.npz', F=F_sevenpoint, M=M, pts1=sevenpoints_1, pts2=sevenpoints_2)
    # print('F from seven point algo: ', F_sevenpoint)
    # print(np.load('../data/q2_2.npz').files)
    # helper.displayEpipolarF(im1, im2, F_sevenpoint)

    # --------------------------------   3.1  -------------------------------------------
    print('----------------  QUESTION 3.1  ---------------')

    # intrinsics = np.load('../data/intrinsics.npz')
    # K1 = intrinsics['K1']
    # K2 = intrinsics['K2']
    #
    # F = sub.eightpoint(pts1, pts2, M)
    # E = sub.essentialMatrix(F, K1, K2)
    # print('F from eight point algo: ', F)
    # print('Eseenetial Matrix from eight point algo: ', E)

    # --------------------------------   3.2 and 3.3  -------------------------------------------
    print('----------------  QUESTION 3.2 and 3.3  ---------------')

    # Done in findM2.py

    # --------------------------------   4.1  -------------------------------------------
    print('----------------  QUESTION 4.1  ---------------')

    # x1 = pts1[:,0]
    # y1 = pts1[:,1]
    #
    # # FIND EPIPOLAR PTS2 CORRESPONDANCES
    # pts1_new = []
    # pts2_new = []
    #
    # F = sub.eightpoint(pts1, pts2, M)
    # # helper.epipolarMatchGUI(im1, im2, F)
    #
    # pts1_GUI = np.asarray( [
    # [126.67419354838708, 210.11367741935487],
    # [70.65698924731183, 128.35883870967746],
    # [480.94516129032263, 95.05131182795697],
    # [494.57096774193553, 152.58249462365586],
    # [515.7666666666668, 229.79539784946235],
    # [427.95591397849466, 132.9007741935484],
    # [329.54731182795706, 231.309376344086],
    # [229.62473118279567, 219.19754838709684],
    # [447.63763440860225, 387.24916129032266],
    # [169.06559139784943, 129.8728172043011],
    # [373.45268817204305, 223.73948387096777]
    # ] )
    #
    # pts2_GUI = np.asarray( [
    # [125, 174],
    # [70, 116],
    # [471, 95],
    # [488, 132],
    # [513, 188],
    # [422, 137],
    # [329, 211],
    # [228, 201],
    # [457, 397],
    # [168, 133],
    # [372, 205]
    # ] )
    #
    # np.savez('../data/q4_1.npz', F=F, pts1=pts1_GUI, pts2=pts2_GUI)
    # print(np.load('../data/q4_1.npz').files)
    # print(np.load('../data/q4_1.npz')['pts1'])
    # print(np.load('../data/q4_1.npz')['pts2'])

    # --------------------------------   4.2  -------------------------------------------
    print('----------------  QUESTION 4.2  ---------------')

    # Done in visualize.py

    # --------------------------------   5.1  -------------------------------------------
    print('----------------  QUESTION 5.1  ---------------')

    # # COMPARISON OF EIGHT POINT AND RANSAC
    # data = np.load('../data/some_corresp_noisy.npz')
    # pts1 = data['pts1'].astype(int)
    # pts2 = data['pts2'].astype(int)
    #
    #
    # # EIGHT POINT
    # F_eight = sub.eightpoint(pts1, pts2, M)
    # P_init, C2_init, M2_init, err = findM2.bestM2(pts1, pts2, F_eight, K1, K2)
    # print('reprojection error for eight point algo: ', err)
    # # helper.displayEpipolarF(im1, im2, F_eight)
    #
    # # RANSAC
    # F_ransac, inliers_best = sub.ransacF(pts1, pts2, M)
    # pts1_inliers= pts1[np.where(inliers_best)]
    # pts2_inliers= pts2[np.where(inliers_best)]
    # P_init, C2_init, M2_init, err = findM2.bestM2(pts1_inliers, pts2_inliers, F_ransac, K1, K2)
    # print('reprojection error after ransac: ', err)
    # # helper.displayEpipolarF(im1, im2, F_ransac)


    # --------------------------------   5.2  -------------------------------------------
    print('----------------  QUESTION 5.2  ---------------')

    '''
    r_test = np.asarray([1,1,1]).astype(float)
    R = np.zeros( (3,3) )
    cv2.Rodrigues(r_test, R, None)
    print(R)
    cv2.Rodrigues(R, r_test, None)
    print(r_test)
    R_init = sub.rodrigues(r_test)
    print(R_init)
    r = sub.invRodrigues(R_init)
    print(r)
    '''



    # M1 = np.array([ [ 1,0,0,0 ],
    #                 [ 0,1,0,0 ],
    #                 [ 0,0,1,0 ]  ])
    #
    # data = np.load('../data/some_corresp_noisy.npz')
    # pts1 = data['pts1'].astype(int)
    # pts2 = data['pts2'].astype(int)
    # M = 640
    # # RANSAC
    # F_ransac, inliers_best = sub.ransacF(pts1, pts2, M)
    # pts1_inliers= pts1[np.where(inliers_best)]
    # pts2_inliers= pts2[np.where(inliers_best)]
    #
    # P_init, C2_init, M2_init, err = findM2.bestM2(pts1_inliers, pts2_inliers, F_ransac, K1, K2)
    # print('error before optimization: ', err)
    #
    # # VISUALIZE INITIAL 3D POINTS
    # visualize.points_3d_visualize(P_init)
    #
    # M2, P = sub.bundleAdjustment(K1, M1, pts1_inliers, K2, M2_init, pts2_inliers, P_init)
    #
    # # CALCULATING REPROJECTION ERROR
    # P_temp, err_opt = sub.triangulate(K1.dot(M1), pts1_inliers, K2.dot(M2), pts2_inliers)
    # print('error after optimization: ', err_opt)
    #
    # # VISUALIZE OPTIMIZED 3D POINTS
    # visualize.points_3d_visualize(P)

