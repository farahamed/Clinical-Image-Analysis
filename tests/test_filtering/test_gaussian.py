import os
import cv2
import numpy as np
from processing.filtering.linear_filters import apply_gaussian



BASE_DIR = os.path.abspath(os.path.join(__file__, "../../.."))
IMAGE_PATH = os.path.join(BASE_DIR, "assets", "test.jpeg")


image = cv2.imread(IMAGE_PATH, 0)

if image is None:
    raise FileNotFoundError(f"Image not found at: {IMAGE_PATH}")


image = cv2.resize(image, (256, 256))

print("Image shape:", image.shape)
print("Running Gaussian Filter...")


result = apply_gaussian(image, 3, 1.0)

cv2.imshow("Original", image)
cv2.imshow("Gaussian Filter", result)

cv2.waitKey(0)
cv2.destroyAllWindows()