import os
import time

import cv2
import numpy as np

from detect import match_keypoints, resize_to_480, detect_and_compute


def run_test(detector, matcher, reference, ref_keypoints, ref_descriptors, path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = resize_to_480(img)

    start = time.time()
    matches, target_keypoints = match_keypoints(detector, matcher, img, ref_descriptors)
    end = time.time()
    print(f'Matched features in {path} in {end - start}s')

    img_matches = np.empty((max(reference.shape[0], img.shape[0]), reference.shape[1] + img.shape[1], 3), dtype=np.uint8)
    cv2.drawMatches(
        reference,
        ref_keypoints,
        img,
        target_keypoints,
        matches,
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
    ref_keypoints, ref_descriptors = detect_and_compute(detector, reference)

    matcher = cv2.DescriptorMatcher.create(cv2.DescriptorMatcher_FLANNBASED)

    for file in os.listdir('./samples/positive'):
        run_test(
            detector,
            matcher,
            reference,
            ref_keypoints,
            ref_descriptors,
            f'./samples/positive/{file}'
        )

    for file in os.listdir('./samples/negative'):
        run_test(
            detector,
            matcher,
            reference,
            ref_keypoints,
            ref_descriptors,
            f'./samples/negative/{file}'
        )


if __name__ == '__main__':
    main()
