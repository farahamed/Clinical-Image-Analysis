import numpy as np
import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)
from processing.frequency.notch_mask import (
    ideal_notch,
    butterworth_notch,
    gaussian_notch
)


def apply_notch_filter(
    fft_shifted,
    points,
    filter_type="ideal",
    radius=20,
    order=2
):

    rows, cols = fft_shifted.shape
    shape = (rows, cols)

    print("Applying notch filter...")
    print("Points:", points)
    print("Radius:", radius)
    print("Type:", filter_type)

    if filter_type == "ideal":

        mask = ideal_notch(shape, points, radius)

    elif filter_type == "butterworth":

        mask = butterworth_notch(shape, points, radius, order)

    elif filter_type == "gaussian":

        mask = gaussian_notch(shape, points, radius)

    else:
        raise ValueError("Unknown filter type")

    filtered_fft = fft_shifted * mask

    return filtered_fft, mask