
"""
    File name: utils.py
    Author: Andrea Costanzo;
"""

import os
import numpy as np
import sys
import cv2
from skimage.util import view_as_windows as viewW
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, cophenet, fcluster
from sklearn.cluster import KMeans


def read_image(file):
    img = cv2.imread(file, -1)
    if img.shape[-1] == 4 and len(img.shape) == 3:
        img = img[:, :, :3]
    return img

def hierarchical_agglomerate_clustering(vec1, vec2, metric='ward', clus_thr=3, depth=4):

    p = np.flip(np.vstack((np.asarray(vec1), np.asarray(vec2))), 1)
    distance = pdist(p)
    Z = linkage(distance, metric)
    c, _ = cophenet(Z, distance)
    return fcluster(Z, t=clus_thr, depth=depth, criterion='inconsistent')


def plot_matches(in_img, kpts_set1, kpts_set2):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(in_img[:, :, ::-1])
    for p in range(0, len(kpts_set1)):
        plt.plot([kpts_set1[p][0], kpts_set2[p][0]], [kpts_set1[p][1], kpts_set2[p][1]], 'r-')
        plt.plot(kpts_set1[p][0], kpts_set1[p][1], 'b.')
        plt.plot(kpts_set2[p][0], kpts_set2[p][1], 'b.')
    plt.axis('off')

    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig('DETECTION_fiori_gialli.png', bbox_inches=extent, transparent=True, pad_inches=0, dpi=230)
    plt.close('all')

    return


def plt_matches_over_clusters(in_img, kpts_set1, kpts_set2, clusters):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(in_img[:, :, ::-1])
    for p in range(0, len(kpts_set1)):
        plt.plot([kpts_set1[p][0], kpts_set2[p][0]], [kpts_set1[p][1], kpts_set2[p][1]], 'r-')
        plt.plot(kpts_set1[p][0], kpts_set1[p][1], 'b.')
        plt.plot(kpts_set2[p][0], kpts_set2[p][1], 'b.')
    plt.axis('off')

    p = np.vstack((np.asarray(kpts_set1), np.asarray(kpts_set2)))

    # Plot points with cluster dependent colors
    plt.scatter(p[:, 0], p[:, 1], c=clusters, cmap='jet')
    plt.axis('off')

    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig('CMFD_clusters.png', bbox_inches=extent, transparent=True, pad_inches=0, dpi=230)
    plt.close('all')

    return


def plt_contours_over_image(in_img, contours):

    # Display the image and plot all contours found
    fig, ax = plt.subplots()
    ax.imshow(in_img, interpolation='nearest', cmap='gray')

    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])

    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig('CMFD_contours.png', bbox_inches=extent, transparent=True, pad_inches=0, dpi=230)
    plt.close('all')

    return


def imfill(bw_im, location=(0, 0)):

    """
    Fill image regions and holes. Performs a flood-fill operation on background pixels of the input binary image,
    starting from the point specified in location.
    :param bw_im: input binary image
    :param location: starting point for flood-filling
    :return: flood-filled image
    """
    im_flood = bw_im.copy()

    # Mask used to flood filling. Note that the size needs to be 2 pixels larger than the image
    h, w = bw_im.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Flood-fill from point location (default is (0, 0))
    cv2.floodFill(im_flood, mask=mask, seedPoint=location, newVal=255)

    # Invert flood-filled image
    im_floodfill_inv = cv2.bitwise_not(im_flood)

    # Combine the two images to get the foreground
    return bw_im | im_floodfill_inv


def im2col(input_im, block_size, stepsize=1):
    """ Rearrange image blocks into columns exactly as Matlab's im2col does
    :param input_im: input image
    :param block_size: size of the image blocks
    :param stepsize: sliding parameter used to select blocks
    :return: reshaped image
    """
    im_copy = input_im.copy()
    return viewW(np.ascontiguousarray(im_copy), block_size).reshape(-1, block_size[0] * block_size[1]).T[:, ::stepsize]


def fast_zncc(original_im, warped_im, corr_window=7, max_subimage_width=400):
    """
        Block-wise correlation measure based on zero mean normalized cross-correlation(ZNCC) between the gray-scale
        of the original image  and its warped version according to the affine transformation estimated by starting from
        SIFT keypoints. Processing id
        :param original_im: original image
        :param warped_im: affine-transformed image
        :param corr_window: kernel size (odd value) for the sliding window-based correlation computation
        :param max_subimage_width: max size of image width sub-blocks for faster processing 
        :return: ZNCC correlation as matrix of the same size of the input image
    """
    assert original_im.shape == warped_im.shape
    assert corr_window % 2 != 0

    # ZNCC is computed on grayscale images
    if len(original_im.shape) == 3:
        original_im = np.float32(cv2.cvtColor(original_im, cv2.COLOR_BGR2GRAY))

    if len(warped_im.shape) == 3:
        warped_im = np.float32(cv2.cvtColor(warped_im, cv2.COLOR_BGR2GRAY))

    # Half kernel size
    border = (corr_window - 1) / 2

    # The image is not processed full-frame but rather sub-divided into large blocks with width determined by the 
    # max_subimage_width parameter
    if max_subimage_width > original_im.shape[1]:
        max_subimage_width = original_im.shape[1]
        
    # Insert the extrema of the sub-images in this array
    blocks = np.append(np.arange(0, warped_im.shape[1], max_subimage_width), warped_im.shape[1])

    # Init output map
    zncc_map = np.zeros(original_im.shape)

    # For each extremum
    for ii in np.arange(0, len(blocks)-1):
        
        # Deal with the usual nuisances related to borders ...
        if ii == 0:
            lmargin = ii
            rmargin = blocks[ii + 1] + border + 1
        elif ii == len(blocks) - 2:
            lmargin = blocks[ii] - border
            rmargin = blocks[ii + 1]
        else:
            lmargin = blocks[ii] - border
            rmargin = blocks[ii + 1] + border

        # Extrapolate current sub-images from full-frame input images
        orig_subimg = original_im[:, int(lmargin):int(rmargin)] / 255.0
        warp_subimg = warped_im[:, int(lmargin):int(rmargin)] / 255.0

        # Rearrange image blocks into columns
        orig_subimg_as_cols = im2col(orig_subimg.transpose(), (corr_window, corr_window))
        warp_subimg_as_cols = im2col(warp_subimg.transpose(), (corr_window, corr_window))

        # Compute averages in the two reshaped blocks
        orig_blk_mean = np.mean(orig_subimg_as_cols, 0)
        warp_blk_mean = np.mean(warp_subimg_as_cols, 0)

        # Normalised Cross-Correlation (NCC) for current blocks
        den = np.sqrt(np.sum((warp_subimg_as_cols - warp_blk_mean)**2, 0) *
                      np.sum((orig_subimg_as_cols - orig_blk_mean)**2, 0))
        den[den == 0] += np.finfo(float).eps
        corr_score = np.sum(((warp_subimg_as_cols - warp_blk_mean) * (orig_subimg_as_cols - orig_blk_mean)) / den, 0)

        # Update the portion of the ZNCC map that corresponds to the initial image crop, deal again with borders
        r_rows = int((warped_im.shape[0] - 2 * border))
        r_cols = int((rmargin - lmargin) - 2 * border)

        zncc_map[int(border):int(original_im.shape[0] - border), int(border + lmargin):int(rmargin - border)] = \
            np.reshape(corr_score.transpose(), (r_cols, r_rows)).transpose()

        zncc_map[zncc_map < 0] = 0

    return zncc_map
