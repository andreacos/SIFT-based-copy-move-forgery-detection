
"""
    File name: localisation.py
    Author: Andrea Costanzo
"""

import cv2
import numpy as np
from detection import copy_move_detector
import ransac
import utils
import itertools
from configuration import Config
from skimage import measure, morphology
from matplotlib import path, pyplot as plt
import scipy.ndimage as ndimage


def copy_move_localisation(img):
    """
    SIFT-based copy-move forgery detection and localisation

    Args:
      img: image as file path or as HxWxC matrix. Algorithms' parameters are loaded from configuration.py
    """

    n_clusters_in_mask = 0
    mask = np.zeros((img.shape[0], img.shape[1]))

    # Detect keypoints and match them
    n_matches, p1, p2, n_kpts = copy_move_detector(img)

    if n_matches == 0:
        return mask, n_clusters_in_mask, 0

    p = np.vstack((p1, p2))

    # Hierarchical Agglomerate Clustering
    clusters = utils.hierarchical_agglomerate_clustering(p1, p2,
                                                         metric=Config().Cmfd.Localizer.hacMetric,
                                                         clus_thr=Config().Cmfd.Localizer.hacMinClusters,
                                                         depth=Config().Cmfd.Localizer.hacMaxDepth)

    if Config().Cmfd.Localizer.saveImages:
        utils.plt_matches_over_clusters(in_img=img, kpts_set1=p1, kpts_set2=p2, clusters=clusters)

    # No copy-move was found
    if np.max(clusters) < Config().Cmfd.Localizer.minClustersForLocalization:
        return mask, n_clusters_in_mask, len(n_kpts)

    # Enumerate all the possible combinations among the number of detected clusters
    n_clust_combinations = list(itertools.combinations(np.arange(np.min(clusters), np.max(clusters)+1), 2))

    # For each pair (C1, C2) of clusters and given the two sets of keypoints (KS1, KS2) matching with eachother
    avgA2 = np.zeros((3, 3))
    for l in range(0, len(n_clust_combinations)):

        z1 = []
        z2 = []

        # Count how many matches fall inside (C1, C2). That is, count how many members of KS1 are in C1 and how many
        # of their corresponding members of KS2 are in C2
        C1 = n_clust_combinations[l][0]
        C2 = n_clust_combinations[l][1]

        # Since KS1 and KS2 are in the same array, this is the offset that allows to pick corresponding matches
        offset = p.shape[0]//2
        for r in range(0, offset):
            if clusters[r] == C1 and clusters[r+offset] == C2:
                z1.append(p[r, :])
                z2.append(p[r+offset, :])

            if clusters[r] == C2 and clusters[r+offset] == C1:
                z1.append(p[r+offset, :])
                z2.append(p[r, :])

        # If there are enough matches in (C1, C2)

        if len(z1) > Config().Cmfd.Localizer.minPointsPerCluster and \
                len(z2) > Config().Cmfd.Localizer.minPointsPerCluster:

            z1 = np.asarray(z1)
            z2 = np.asarray(z2)

            h, inliers = ransac.ransacfithomography(z1, z2)

            x1p, T1, _ = ransac.normalise2d(z1[inliers.flatten()])
            x2p, T2, _ = ransac.normalise2d(z2[inliers.flatten()])
            A = ransac.vgg_affine(h, x1p, x2p)
            A = np.linalg.lstsq(T2, A, rcond=None)[0].dot(T1)

            # Repeat from C2 to C1 with inverse affine transform
            A2 = ransac.vgg_affine(h, x2p, x1p)
            A2 = np.linalg.lstsq(T1, A2, rcond=None)[0].dot(T2)

            avgA2 += A2

            # Regions with area less than the following value are discarded
            min_blob_area = int(Config().Cmfd.Localizer.minBlobAreaAsPercentOfImgSize * img.shape[0] * img.shape[1])

            # Mask direct
            mask = mask + localise_tampering(in_img=img, Amatrix=A, x1=z1, x2=z2,
                                             min_blobArea=min_blob_area,
                                             th_bin_mask=Config().Cmfd.Localizer.znccMinBinaryThreshold)

            # Mask inverse
            mask = mask + localise_tampering(in_img=img, Amatrix=A2, x1=z1, x2=z2,
                                             min_blobArea=min_blob_area,
                                             th_bin_mask=Config().Cmfd.Localizer.znccMinBinaryThreshold)

            # Update the number of clusters included in mask
            n_clusters_in_mask += 1

    return mask, n_matches, n_clusters_in_mask


def localise_tampering(in_img, Amatrix, x1, x2, min_blobArea, th_bin_mask):
    """
    Tampering localisation algorithm

    Args:
        in_img: image as HxWxC matrix
        Amatrix: 3x3 homography transformation matrix
        x1: first set of keypoints
        x2: second set of keypoints
        min_blobArea: minimum area for a detected region
        th_bin_mask: binarisation threshold for ZNCC algorithm
    Returns:
        mask: tampering mask (0 = original, 1 = tampered)
    """

    # Compute ZNCC score between the original image and its warped version according to the affine transformation
    # estimated from keypoints clusters. This step allows to find the cloned regions underlying keypoint clusters
    warped_im = cv2.warpPerspective(in_img, Amatrix, (in_img.shape[1], in_img.shape[0]))
    zncc_map = utils.fast_zncc(original_im=in_img,
                               warped_im=warped_im,
                               corr_window=Config().Cmfd.Localizer.znccKernelSize,
                               max_subimage_width=Config().Cmfd.Localizer.znccMaxSubimageWidth)

    # Gaussian smoothing of ZNNC score to reduce noise
    zncc_map = np.uint8(255 * (zncc_map - np.min(zncc_map)) / (np.max(zncc_map) - np.min(zncc_map)))
    zncc_map_smooth = cv2.GaussianBlur(zncc_map, (7, 7), sigmaX=0.5, sigmaY=0.5)

    # Binary thresholding of the map
    _, zncc_bin = cv2.threshold(np.uint8(zncc_map_smooth)/255., th_bin_mask, 1, cv2.THRESH_BINARY)

    # Fill holes in the binary mask (cast as boolean) and remove regions that are too small
    im = ndimage.binary_fill_holes(True * zncc_bin)
    im = morphology.remove_small_objects(im, min_size=min_blobArea, connectivity=2)

    # Find all contours around white blobs
    contours = measure.find_contours(im, 0.8)

    in_model_x = np.hstack((x1[:, 0], x2[:, 0]))
    in_model_y = np.hstack((x1[:, 1], x2[:, 1]))

    pts_to_check = np.hstack((np.expand_dims(np.array(in_model_y), 1), np.expand_dims(np.array(in_model_x), 1)))

    mask = np.zeros_like(im, dtype='bool')

    # For each contour, determine the set of points inside it and check if key-points belong to it
    for contour in contours:
        p = path.Path(contour)
        pts_in_p = p.contains_points(pts_to_check)
        if np.any(pts_in_p):

            # Draw contour on the mask by using coordinates rounded to their nearest integer value
            mask[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')] = 1

            # Fill in the holes created by the contour boundary
            mask = ndimage.binary_fill_holes(mask)

    return np.uint8(mask > 0)


if __name__ == '__main__':

    image = cv2.imread('assets/fiori-gialli.jpg')
    mask, _, _ = copy_move_localisation(image)
    bin_mask = 255 * (np.uint8(mask > 0))

    plt.imsave('MASK_fiori_gialli.png', bin_mask, cmap='gray')
    plt.imsave('CMFD_fiori_gialli.png', np.tile(np.expand_dims(mask, 2) > 0, (1, 1, 3)) * image[:, :, ::-1])
