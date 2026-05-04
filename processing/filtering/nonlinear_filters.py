import numpy as np
from .convolution import zero_pad


def median_filter(image, size):
    image = image.astype(np.float64)

    h, w = image.shape
    k = size // 2

     #padded = zero_pad(image, k, k)

    padded = np.pad(image, ((k, k), (k, k)), mode='constant')

    result = np.zeros((h, w), dtype=np.float64)

    for i in range(h):
        for j in range(w):
            window = padded[i:i+size, j:j+size].flatten()
            window.sort()
            result[i, j] = window[len(window)//2]

    return result
 