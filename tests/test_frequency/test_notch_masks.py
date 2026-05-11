import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

import matplotlib.pyplot as plt

from processing.frequency.notch_mask import (
    ideal_notch,
    butterworth_notch,
    gaussian_notch
)


shape = (512, 512)

points = [
    (180, 250)
]

radius = 20


ideal = ideal_notch(shape, points, radius)

butter = butterworth_notch(
    shape,
    points,
    radius,
    order=2
)

gaussian = gaussian_notch(
    shape,
    points,
    radius
)


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Ideal Notch")
plt.imshow(ideal, cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Butterworth Notch")
plt.imshow(butter, cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Gaussian Notch")
plt.imshow(gaussian, cmap='gray')

plt.show()