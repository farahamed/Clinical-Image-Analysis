import os
import cv2
import numpy as np
from processing.filtering.linear_filters import apply_average


def load_test_image():
    BASE_DIR = os.path.abspath(os.path.join(__file__, "../../.."))
    IMAGE_PATH = os.path.join(BASE_DIR, "assets", "test.jpeg")

    image = cv2.imread(IMAGE_PATH, 0)

    if image is None:
        raise FileNotFoundError(f"Could not load image at: {IMAGE_PATH}")

    # Resize for consistency and speed
    image = cv2.resize(image, (256, 256))

    return image


image = load_test_image()

print("Running Average Filter...")

result = apply_average(image, 3)


cv2.imshow("Original", image)
cv2.imshow("Average Filter", result)

cv2.waitKey(0)
cv2.destroyAllWindows()