
"""
    File name: detection.py
    Author: Andrea Costanzo
"""

import cv2
import numpy as np
from utils import plot_matches
from configuration import Config


def copy_move_detector(img):
    """
    SIFT-based copy-move forgery detector
    Args:
      img: image as file path or as HxWxC matrix. Algorithm parameters are loaded from configuration.py

    """

    # Load image, convert to grayscale
    if isinstance(img, str):
        gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Extract SIFT keypoints
    sift = cv2.SIFT_create(contrastThreshold=Config().Sift.contrastThreshold,
                           edgeThreshold=Config().Sift.edgeThreshold,
                           nfeatures=Config().Sift.nFeatures,
                           nOctaveLayers=Config().Sift.nOctaveLayers,
                           sigma=Config().Sift.sigma)

    (kps, descr) = sift.detectAndCompute(gray, None)
    # print('Found {} key points'.format(len(kps)))

    descr = np.asarray(descr)
    z = np.dot(descr, descr.transpose())
    d = np.tile(np.sqrt(np.diag(z)), (descr.shape[1], 1)).transpose()
    descr = descr / (d + np.finfo(float).eps)
    z = np.dot(descr, descr.transpose())

    p1 = []
    p2 = []

    is_matched = [False]*len(kps)

    # For each keypoint start looking for a match
    for i in range(0, len(kps)):
        dotprods = z[i, :]
        dotprods[dotprods > 1] = 1
        vals = np.sort(np.arccos(dotprods))
        idxs = np.argsort(np.arccos(dotprods))

        # Evaluate three conditions:
        # (1) Distance of current point from itself is small
        # (2) Distance with 3rd point in list is significantly higher than distance from the 2nd, supposedly the match
        # (3) Current point is not yet matched with any other point
        if vals[0] < Config().Cmfd.Detector.minDistanceFromSelf and \
                vals[1]/(vals[2] + np.finfo(float).eps) < Config().Cmfd.Detector.minDistanceFromTop2Ratio and \
                not is_matched[idxs[1]]:

            # Update matched status so that these keypoints are not considered anymore
            is_matched[idxs[1]] = True
            is_matched[i] = True

            # Get the coordinates of the two keypoints and evaluate their distance: if distant enough, match is kept
            kp_current = kps[i].pt
            kb_best_match = kps[idxs[1]].pt
            pdist = np.linalg.norm(np.asarray(kp_current) - np.asarray(kb_best_match))    # Euclidean distance
            if pdist > Config().Cmfd.Detector.minDistanceFromMatchingPoints:
                p1.append(kp_current)
                p2.append(kb_best_match)

    # print('Found {} matches'.format(len(p1)))

    # Save input image with overlaid matches as PNG
    if Config().Cmfd.Detector.saveImages:
        plot_matches(in_img=img, kpts_set1=p1, kpts_set2=p2)

    return len(p1), np.array(p1), np.array(p2), len(kps)


if __name__ == '__main__':

    im = cv2.imread("assets/fiori-gialli.jpg")
    n_matches = copy_move_detector(im)
