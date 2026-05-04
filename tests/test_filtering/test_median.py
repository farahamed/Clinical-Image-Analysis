import os
import numpy as np
import cv2
import time

from processing.filtering.nonlinear_filters import median_filter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
IMAGE_PATH = os.path.join(BASE_DIR, "assets", "test.jpeg")

image = cv2.imread(IMAGE_PATH, 0)

if image is None:
    raise FileNotFoundError(f"Could not load image at: {IMAGE_PATH}")



def normalize(img):
    img = img.astype(np.float64)

    min_val = img.min()
    max_val = img.max()

    if max_val - min_val == 0:
        return np.zeros_like(img, dtype=np.uint8)

    img = (img - min_val) / (max_val - min_val)
    img = img * 255

    return img.astype(np.uint8)



print("Running Median Filter...")

start = time.perf_counter()
med = median_filter(image, 3)
end = time.perf_counter()

print(f"Median filter time: {end - start:.4f} seconds")

noisy = image.copy() 
noisy[::10, ::10] = 255

print("Original min/max:", image.min(), image.max())
print("Noisy min/max:", noisy.min(), noisy.max())
print("Median min/max:", med.min(), med.max())


med_display = normalize(med)


cv2.imshow("Original", image)
cv2.imshow("Noisy image",noisy)
cv2.imshow("Median Filter (Normalized)", med_display)

cv2.waitKey(0)
cv2.destroyAllWindows()