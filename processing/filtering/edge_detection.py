import numpy as np
from .convolution import convolve2d, normalize_image


SOBEL_X = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

SOBEL_Y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
], dtype=np.float32)


def apply_edge_detection(image, use_fast_magnitude=True):

    grad_x = convolve2d(image, SOBEL_X)
    grad_y = convolve2d(image, SOBEL_Y)

    if use_fast_magnitude:

        # Faster approximation
        magnitude = np.abs(grad_x) + np.abs(grad_y)

    else:

        magnitude = np.sqrt(grad_x**2 + grad_y**2)

    grad_x = normalize_image(np.abs(grad_x))
    grad_y = normalize_image(np.abs(grad_y))
    magnitude = normalize_image(magnitude)

    return grad_x, grad_y, magnitude