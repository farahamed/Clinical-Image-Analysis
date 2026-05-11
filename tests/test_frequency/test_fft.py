import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

import cv2
import numpy as np
import matplotlib.pyplot as plt

from processing.frequency.fft_utils import (
    compute_fft,
    inverse_fft
)


image = cv2.imread("assets/test.jpeg", 0)

if image is None:
    raise ValueError("Could not load image")


F = compute_fft(image)

reconstructed = inverse_fft(F)


difference = np.abs(
    image.astype(np.float32)
    - reconstructed.astype(np.float32)
)

print("Maximum reconstruction error:", difference.max())


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Reconstructed")
plt.imshow(reconstructed, cmap='gray')

plt.show()