import os
import time

import cv2
import numpy as np


def resize_to_480(img):
    w, h = img.shape
    scale = 480 / h
    return cv2.resize(img, (480, int(w * scale)))


def match_and_display(detector, matcher, reference, keypoints1, descriptors1, path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = resize_to_480(img)

    start = time.time()
    keypoints2, descriptors2 = detector.detectAndCompute(img, None)

    # Cast descriptors to F32 for FLANN
    descriptors2 = descriptors2.astype(np.float32, copy=False)
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)

    ratio_thresh = 0.7
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    end = time.time()

    print(f'Matched features in {path} in {end - start}s')

    img_matches = np.empty((max(reference.shape[0], img.shape[0]), reference.shape[1] + img.shape[1], 3), dtype=np.uint8)
    cv2.drawMatches(
        reference,
        keypoints1,
        img,
        keypoints2,
        good_matches,
        # np.array(knn_matches)[:, 0],
        img_matches,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imshow('Matches', img_matches)
    cv2.waitKey()


def main():
    reference = cv2.imread('./samples/reference.png', cv2.IMREAD_GRAYSCALE)
    reference = resize_to_480(reference)

    cv2.namedWindow('Matches', cv2.WINDOW_NORMAL)

    detector = cv2.ORB.create()
    keypoints1, descriptors1 = detector.detectAndCompute(reference, None)
    descriptors1 = descriptors1.astype(np.float32, copy=False)

    matcher = cv2.DescriptorMatcher.create(cv2.DescriptorMatcher_FLANNBASED)

    for file in os.listdir('./samples/positive'):
        match_and_display(detector, matcher, reference, keypoints1, descriptors1, f'./samples/positive/{file}')

    for file in os.listdir('./samples/negative'):
        match_and_display(detector, matcher, reference, keypoints1, descriptors1, f'./samples/negative/{file}')


if __name__ == '__main__':
    main()
