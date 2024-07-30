import cv2
import numpy as np


def resize_to_480(img):
    w, h = img.shape
    scale = 480 / h
    return cv2.resize(img, (480, int(w * scale)))


def detect_and_compute(detector, img):
    keypoints, descriptors = detector.detectAndCompute(img, None)

    # Cast descriptors to F32 for FLANN
    descriptors = descriptors.astype(np.float32, copy=False)
    return keypoints, descriptors


def match_keypoints(detector, matcher, img, ref_descriptors):
    target_keypoints, target_descriptors = detect_and_compute(detector, img)
    knn_matches = matcher.knnMatch(ref_descriptors, target_descriptors, 2)

    ratio_thresh = 0.7
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    return good_matches, target_keypoints
