import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)
import cv2
import matplotlib.pyplot as plt

from processing.frequency.fft_utils import compute_fft
from processing.frequency.spectrum_display import magnitude_spectrum
from processing.frequency.notch_filters import apply_notch_filter
from processing.frequency.reconstruction import reconstruct_image


image = cv2.imread("assets/test.jpeg", 0)

if image is None:
    raise ValueError("Could not load image")


# FFT
F = compute_fft(image)

# Spectrum
spectrum = magnitude_spectrum(F)

# Simulated clicked points
points = [
    (180, 250)
]

# Apply notch
filtered_fft, mask = apply_notch_filter(
    F,
    points,
    filter_type='gaussian',
    radius=15
)

# Reconstruct
result = reconstruct_image(filtered_fft)


plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.title("Original")
plt.imshow(image, cmap='gray')

plt.subplot(2, 2, 2)
plt.title("Spectrum")
plt.imshow(spectrum, cmap='gray')

plt.subplot(2, 2, 3)
plt.title("Notch Mask")
plt.imshow(mask, cmap='gray')

plt.subplot(2, 2, 4)
plt.title("Filtered Result")
plt.imshow(result, cmap='gray')

plt.show()