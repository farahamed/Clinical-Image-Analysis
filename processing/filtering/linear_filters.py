import numpy as np
from .convolution import convolve2d


GAUSSIAN_CACHE = {}
AVERAGE_CACHE = {}


def average_kernel(size):

    if size in AVERAGE_CACHE:
        return AVERAGE_CACHE[size]

    kernel = np.ones((size, size), dtype=np.float32)
    kernel /= (size * size)

    AVERAGE_CACHE[size] = kernel

    return kernel


def gaussian_kernel(size, sigma):

    key = (size, sigma)

    if key in GAUSSIAN_CACHE:
        return GAUSSIAN_CACHE[key]

    k = size // 2

    x = np.arange(-k, k + 1)
    y = np.arange(-k, k + 1)

    xx, yy = np.meshgrid(x, y)

    kernel = np.exp(
        -(xx**2 + yy**2) / (2 * sigma**2)
    )

    kernel /= np.sum(kernel)

    kernel = kernel.astype(np.float32)

    GAUSSIAN_CACHE[key] = kernel

    return kernel


def apply_average(image, size):

    kernel = average_kernel(size)

    return convolve2d(image, kernel)


def apply_gaussian(image, size, sigma):

    kernel = gaussian_kernel(size, sigma)

    return convolve2d(image, kernel)