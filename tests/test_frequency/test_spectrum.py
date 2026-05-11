import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

import cv2
import matplotlib.pyplot as plt

from processing.frequency.fft_utils import compute_fft
from processing.frequency.spectrum_display import magnitude_spectrum


image = cv2.imread("assets/test.jpeg", 0)

if image is None:
    raise ValueError("Could not load image")


F = compute_fft(image)

spectrum = magnitude_spectrum(F)


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("FFT Magnitude Spectrum")
plt.imshow(spectrum, cmap='gray')

plt.show()