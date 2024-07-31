# stop-sign-detection

To run:
- Place images containing stop signs in `./samples/positive` (some have already been provided).
- Place images *not* containing stop signs in `./samples/negative` (some have already been provided).
- Run `test.py` to iterate through each image and see what features the algorithm detects!

### About
The idea behind this project is loosely based around [this OpenCV guide about FLANN-based feature matching](https://docs.opencv.org/4.x/d5/d6f/tutorial_feature_flann_matcher.html),
using OpenCV's [ORB](https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html) keypoint detector instead of the SURF
detection used in the guide for licensing reasons.

Do stop signs naturally lend themselves to feature tracking? Not *really*. But there are seemingly enough features in
the stop sign text to differentiate stop-sign photos from non stop-sign photos with decent accuracy.

### Alternatives to consider
- Using an OCR algorithm like [tesseract](https://pypi.org/project/pytesseract/) to detect the word "stop" in each image,
  though it will take a long time to run (probably insufficient for RACECAR purposes).
- Training a CNN on stop sign images; this will likely be faster due to the TPU on the racecars.
