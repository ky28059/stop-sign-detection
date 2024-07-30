import cv2
import numpy as np

img1 = cv2.imread('./samples/positive/IMG_2674.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('./samples/positive/IMG_2678.jpg', cv2.IMREAD_GRAYSCALE)

detector = cv2.ORB.create()
keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
keypoints2, descriptors2 = detector.detectAndCompute(img2, None)

# Cast descriptors to F32 for FLANN
descriptors1 = descriptors1.astype(np.float32, copy=False)
descriptors2 = descriptors2.astype(np.float32, copy=False)

matcher = cv2.DescriptorMatcher.create(cv2.DescriptorMatcher_FLANNBASED)
knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)

ratio_thresh = 0.7
good_matches = []
for m, n in knn_matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)

img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
cv2.drawMatches(
    img1,
    keypoints1,
    img2,
    keypoints2,
    good_matches,
    # np.array(knn_matches)[:, 0],
    img_matches,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.namedWindow('Matches', cv2.WINDOW_NORMAL)
cv2.imshow('Matches', img_matches)
cv2.waitKey()
