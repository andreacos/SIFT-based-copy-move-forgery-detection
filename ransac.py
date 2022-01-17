
"""
    File name: ransac.py
    Author: Andrea Costanzo
"""

import numpy as np


def ransacfithomography(x1, x2, min_pts=4):

    assert x1.shape == x2.shape and min_pts >= 4 and 2 <= x1.shape[1] == x2.shape[1] <= 3

    # Normalise each set of points
    x1, T1, c1 = normalise2d(x1)
    x2, T2, c2 = normalise2d(x2)

    H, inliers = ransac(x1, x2, min_pts)

    if inliers.shape[1] > 3:
        xin1 = x1[inliers.flatten()]
        xin2 = x2[inliers.flatten()]
        A = vgg_affine(H, xin1, xin2)

        # Denormalise
        A = np.dot(np.linalg.solve(T2, A), T1)

    return A, inliers


def homogdist2d(H, x1, x2, thr):

    x1 = np.concatenate((x1, np.ones((x1.shape[0], 1))), 1)
    x2 = np.concatenate((x2, np.ones((x2.shape[0], 1))), 1)

    Hx1 = np.dot(H, x1.transpose())
    invHx2 = np.linalg.lstsq(H, x2.transpose(), rcond=None)[0]

    def hnormalise(x):
        nx = x
        for k in range(0, x.shape[1]):
            nx[:, k] = np.divide(x[:, k], x[:, 2])
        return nx

    x1 = hnormalise(x1)
    x2 = hnormalise(x2)
    Hx1 = hnormalise(Hx1.transpose())
    invHx2 = hnormalise(invHx2.transpose())

    d2 = np.sum((x1 - np.asarray(invHx2))**2, 1) + np.sum((x2 - np.asarray(Hx1))**2, 1)

    inliers = np.where(d2 < thr)

    return np.asarray(inliers), H


def homography2d(x1, x2):

    x1, T1, _ = normalise2d(x1)
    x2, T2, _ = normalise2d(x2)

    x1 = np.concatenate((x1, np.ones((x1.shape[0], 1))), 1)
    x2 = np.concatenate((x2, np.ones((x2.shape[0], 1))), 1)

    npts = x1.shape[0]

    A = np.zeros((1, 9))
    O = np.asarray([0]*3)

    for n in range(0, npts):
        X = x1[n, :]
        x = x2[n, 0]
        y = x2[n, 1]
        w = x2[n, 2]

        a1 = np.concatenate((O, -w*X, y*X), 0).reshape(1, 9)
        a2 = np.concatenate((w*X, O, -x*X), 0).reshape(1, 9)
        a3 = np.concatenate((-y*X, x*X, O), 0).reshape(1, 9)

        A = np.concatenate((A, a1, a2, a3), 0)

    A = A[1:, :]

    _, _, V = np.linalg.svd(A)
    H = V[-1, :].reshape(3, 3)

    # De-normalise
    return np.linalg.lstsq(T2, H, rcond=None)[0].dot(T1)


def iscolinear(p_1, p_2, p_3, coord='inhomog'):

    if coord == 'homog':
        r = np.abs(np.dot(np.cross(p_1, p_2), p_3)) < np.finfo(float).eps
    elif coord == 'inhomog':
        r = np.linalg.norm(np.cross(p_2-p_1, p_3-p_1)) < np.finfo(float).eps
    return r


def isdegenerate(x, coords='inhomog'):

    y = np.concatenate((x, np.ones((x.shape[0], 1))), 1)

    check = iscolinear(y[0, :], y[1, :], y[2, :]) or iscolinear(y[0, :], y[1, :], y[3, :]) or \
        iscolinear(y[0, :], y[2, :], y[3, :]) or iscolinear(y[1, :], y[2, :], y[3, :])

    return check


def normalise2d(pts):
    """
    Translate and normalise a set of 2D homogeneous points so that their centroid is at the origin and their
    mean distance from the origin is sqrt(2). This process typically improves the conditioning of any equations used to
    solve homographies, fundamental matrices etc.

    Args:
        pts: array of 2D homogeneous coordinates
    """

    cnt = np.mean(pts, 0)
    mean_dist = np.mean(np.sqrt(np.sum((pts-cnt) ** 2, 1)))

    if mean_dist == 0:
        scale = 1.0
    else:
        scale = np.sqrt(2)/mean_dist

    T = np.array([[scale, 0, -scale*cnt[0]],
                  [0, scale, -scale*cnt[1]],
                  [0, 0, 1]])

    # Pad data with homogeneous scale factor of 1
    y = np.concatenate((pts, np.ones((pts.shape[0], 1))), 1).transpose()

    return np.matmul(T, y).transpose()[:, :-1], T, cnt


def ransac(x1, x2, min_pts=4, dist_thr=0.05, max_data_trials=100, max_trials=1000, prob=0.99):

    assert x1.shape == x2.shape and min_pts >= 4 and 0 <= prob <= 1.0

    bestM = np.nan
    trialcount = 0
    bestscore = 0
    N = 1

    while N > trialcount:
        degenerate = True
        count = 1
        M = homography2d(x1, x2)

        # Select random datapoints to form a trial model, M. In selecting these points we have to check that they
        # are not in a degenerate configuration
        while degenerate:
            ind = np.random.choice(x1.shape[0], (min_pts, 1), replace=False).reshape(min_pts)
            degenerate = (isdegenerate(x1[ind, :]) or isdegenerate(x2[ind, :]))

            if not degenerate:
                s1 = x1[ind, :]
                s2 = x2[ind, :]
                M = homography2d(s1, s2)

            count += 1
            if count > max_data_trials:
                print('Unable to select a non-degenerate data set')
                break

        inliers, M = homogdist2d(M, x1, x2, dist_thr)
        n_inliers = inliers.shape[1]

        if n_inliers > bestscore:
            bestscore = n_inliers
            bestinliers = inliers
            bestM = M

            fracinliers = n_inliers / x1.shape[0]
            pNoOutliers = 1 - fracinliers ** min_pts
            pNoOutliers = np.maximum(np.finfo(float).eps, pNoOutliers)
            pNoOutliers = np.minimum(1-np.finfo(float).eps, pNoOutliers)

            N = np.log(1 - prob) / np.log(pNoOutliers)

        trialcount += 1
        if trialcount > max_trials:
            print('Ransac reached the maximum number of {} trials'.format(max_trials))
            break

        if not np.any(np.isnan(bestM)):
            M = bestM
            inliers = bestinliers

    return bestM, bestinliers


def vgg_affine(H, xin, yin):

    assert xin.shape == yin.shape

    xin = np.concatenate((xin, np.ones((xin.shape[0], 1))), 1)
    yin = np.concatenate((yin, np.ones((yin.shape[0], 1))), 1)

    def vgg_not_homog(a):
        return np.divide(a[:, 0:2], np.tile(a[:, 2], (2, 1)).transpose())

    def vgg_condition_2d(pnt, C):

        assert pnt.shape[1] in [2, 3]

        if pnt.shape[1] == 3:
            pc = np.dot(C, pnt.transpose()).transpose()
        else:
            nh = vgg_not_homog(pnt)
            pc = np.dot(C, nh.transpose()).transpose()
            pc = vgg_not_homog(pc)

        return pc

    nonhomog = vgg_not_homog(xin)
    means = np.mean(np.round(nonhomog, 4), 0)
    maxstd = np.max(np.std(nonhomog, axis=0, ddof=1), 0)
    C1 = np.diag([1.0/maxstd, 1.0/maxstd, 1.0])
    C1[:, 2] = np.asarray([-means[0]/maxstd, -means[1]/maxstd, 1.0]).transpose()

    nonhomog_y = vgg_not_homog(yin)
    means_y = np.mean(np.round(nonhomog_y, 4), 0)
    C2 = C1
    C2[:, 2] = np.asarray([-means_y[0] / maxstd, -means_y[1] / maxstd, 1.0]).transpose()

    xin = vgg_condition_2d(xin, C1)
    yin = vgg_condition_2d(yin, C2)

    xin_nh = vgg_not_homog(xin)
    yin_nh = vgg_not_homog(yin)

    A = np.concatenate((xin_nh, yin_nh), 1)

    [_, s, v] = np.linalg.svd(A)

    nullspace_dimension = np.sum(s < 1e3 * s[1] * np.finfo(float).eps)

    if nullspace_dimension > 2:
        print('Warning nullspace')

    B1 = v.transpose()[:2, :2]
    B2 = v.transpose()[2:4, :2]

    conc = np.concatenate((np.dot(B2, np.linalg.pinv(B1)), np.zeros((2, 1))), 1)
    conc = np.concatenate((conc, np.asarray([[0, 0, 1]])), 0)

    Hc = np.dot(np.dot(np.linalg.inv(C2), conc), C1)
    Hc /= Hc[2, 2]

    return Hc
