import numpy as np


def zero_pad(image, pad_h, pad_w):
    h, w = image.shape

    padded = np.zeros((h + 2 * pad_h, w + 2 * pad_w), dtype=image.dtype)

    for i in range(h):
        for j in range(w):
            padded[i + pad_h, j + pad_w] = image[i, j]

    return padded


def convolve2d(image, kernel):
    image = image.astype(np.float64)
    
    h, w = image.shape
    kh, kw = kernel.shape

    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    result = np.zeros((h, w), dtype=np.float64)

    for i in range(h):
        for j in range(w):
            region = padded[i:i + kh, j:j + kw]
            result[i, j] = np.sum(region * kernel)

    return result