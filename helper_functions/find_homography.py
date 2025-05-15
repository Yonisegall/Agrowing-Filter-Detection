import cv2
import numpy as np


def find_homography(grayscale1, grayscale2):
    grayscale1_8bit = cv2.normalize(grayscale1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    grayscale2_8bit = cv2.normalize(grayscale2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    # Create SIFT detector (SIFT is now available in OpenCV)
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(grayscale1_8bit, None)
    kp2, des2 = sift.detectAndCompute(grayscale2_8bit, None)

    # Use FLANN-based matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Find two nearest neighbors for each descriptor
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test to keep good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Check if enough matches are found
    if len(good_matches) < 10:
        raise Exception(f"Not enough matches found: {len(good_matches)}/10")

    # Extract the matched keypoints coordinates
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute the homography matrix using RANSAC to handle outliers
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    return M, mask