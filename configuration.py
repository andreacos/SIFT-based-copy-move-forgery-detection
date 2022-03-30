
"""
    File name: configuration.py
    Author: Andrea Costanzo
"""


# SIFT detector parameters
class Sift:
    nFeatures = 5000                               # Maximum number of keypoints extracted (top-score ones)
    contrastThreshold = 0.03                        # Sift contrast threshold
    edgeThreshold = 10                              # Sift edge threshold
    nOctaveLayers = 3                               # Layers for each octave
    sigma = 1.6                                     # Standard deviation of the Gaussian smoothing filter


# Copy-Move Forgery Detector parameters
class Detector:
    minDistanceFromSelf = 0.01                      # Distance below which a keypoint is matching with itself
    minDistanceFromTop2Ratio = 0.55                 # Threshold for ratio between 2nd, 3rd best matching keypoints
    minDistanceFromMatchingPoints = 10              # Minimum distance between two matching keypoints
    minMatchesForTampering = 4                      # Minimum number of matches to decide on presence of copy-move
    saveImages = True                               # Save intermediate and final results as images


# Copy-Move Forgery Localizer parameters
class Localizer:
    hacMetric = 'ward'                              # Metric for Hierarchical Agglomerate Clustering (HAC)
    hacMinClusters = 3                              # Minimum number of clusters for HAC
    hacMaxDepth = 4                                # Depth for HAC
    minClustersForLocalization = 1                  # Minimum number of clusters for copy-move localization
    minPointsPerCluster = 5                         # Minimum number of points in clusters for copy-move localization
    minBlobAreaAsPercentOfImgSize = 0.01            # Minimum area (as % of total image size) for a detected region
    znccMinBinaryThreshold = 0.28                   # Binarisation threshold for ZNCC algorithm
    znccKernelSize = 7                              # Size of the sliding window used to compute ZNNC score
    znccMaxSubimageWidth = 150                      # Width of the sub-images into with input image is split for ZNCC
    saveImages = True                               # Save intermediate and final results as images


# General wrapper for all Copy-Move detection parameters
class Cmfd:
    Detector = Detector()
    Localizer = Localizer()


# General wrapper for all system parameters
class Config:
    Sift = Sift()
    Cmfd = Cmfd()
