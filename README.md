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

<p align="center">
    <img src="https://github.com/user-attachments/assets/59bdbb8f-9846-43f9-aee5-211597783231" width="400px"> <img src="https://github.com/user-attachments/assets/320f4fa1-7158-40ff-9b30-f8ae241c6514" width="400px">
</p>
<p align="center">
    <img src="https://github.com/user-attachments/assets/ce35a159-eba3-4e90-8232-25cb0a45c313" width="400px"> <img src="https://github.com/user-attachments/assets/616a5025-41e0-4c99-ab69-f9dd587f1838" width="400px">
</p>

### Alternatives to consider
- Using an OCR algorithm like [tesseract](https://pypi.org/project/pytesseract/) to detect the word "stop" in each image,
  though it will take a long time to run (probably insufficient for RACECAR purposes).
- Training a CNN on stop sign images; this will likely be faster due to the TPU on the racecars.
