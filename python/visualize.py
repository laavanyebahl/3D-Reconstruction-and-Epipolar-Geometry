'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''

import numpy as np
import submission as sub
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import findM2


def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)

def points_3d_visualize(P_best):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')

    X = P_best[:,0]
    Y = P_best[:,1]
    Z = P_best[:,2]

    ax.scatter(X, Y, Z, s = 1.7)

    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()

if __name__ == '__main__':

    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')

    # LOAD SAMPLE PTS1
    data = np.load('../data/templeCoords.npz')
    x1 = data['x1'].astype(int).flatten()
    y1 = data['y1'].astype(int).flatten()

    # LOAD F
    q2_1 = np.load('../data/q2_1.npz')
    F = q2_1['F']

    # LOAD INTRINSICS
    intrinsics = np.load('../data/intrinsics.npz')
    K1 = intrinsics['K1']
    K2 = intrinsics['K2']

    M1 = np.array([ [ 1,0,0,0 ],
                    [ 0,1,0,0 ],
                    [ 0,0,1,0 ]  ])

    C1 = K1.dot(M1)

    # FIND EPIPOLAR PTS2 CORRESPONDANCES
    pts1_new = []
    pts2_new = []

    for i in range(x1.shape[0]):
        x2, y2 = sub.epipolarCorrespondence(im1, im2, F, x1[i], y1[i])
        if x2 is not None:
            pts1_new.append([ x1[i], y1[i] ])
            pts2_new.append([ x2, y2 ])

    pts1_new = np.asarray(pts1_new)
    pts2_new = np.asarray(pts2_new)

    # print('pts1_new shape: ', pts1_new.shape)
    # print('pts2_new shape: ', pts2_new.shape)

    # FIND 3D POINTS USING TRIANGULATION
    P_best, C2_best, M2_best, err_best = findM2.bestM2(pts1_new, pts2_new, F, K1, K2)

    np.savez('../data/q4_2.npz', F=F, M1=M1, M2=M2_best, C1=C1 , C2=C2_best )
    print(np.load('../data/q4_2.npz').files)

    ## 3D PLOT RECONSTRUCTION
    points_3d_visualize(P_best)
