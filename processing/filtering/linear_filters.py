import numpy as np
from .convolution import convolve2d


def average_kernel(size):
    kernel = np.ones((size, size), dtype=np.float64)
    return kernel / (size * size)


def gaussian_kernel(size, sigma):
    k = size // 2
    kernel = np.zeros((size, size), dtype=np.float64)

    for x in range(-k, k+1):
        for y in range(-k, k+1):
            value = (1 / (2 * np.pi * sigma**2)) * \
                    np.exp(-(x**2 + y**2) / (2 * sigma**2))
            kernel[x + k, y + k] = value

   
    kernel /= np.sum(kernel)

    return kernel


def apply_average(image, size):
    kernel = average_kernel(size)
    return convolve2d(image, kernel)


def apply_gaussian(image, size, sigma):
    kernel = gaussian_kernel(size, sigma)
    return convolve2d(image, kernel)