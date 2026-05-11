import numpy as np


def zero_pad(image, pad_h, pad_w, mode='edge'):
    return np.pad(
        image,
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode=mode
    )


def convolve2d(image, kernel, padding='edge'):

    image = image.astype(np.float32)
    kernel = kernel.astype(np.float32)

    h, w = image.shape
    kh, kw = kernel.shape

    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(
        image,
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode=padding
    )

    result = np.zeros((h, w), dtype=np.float32)

    for i in range(h):

        rows = padded[i:i + kh]

        for j in range(w):

            region = rows[:, j:j + kw]

            result[i, j] = np.sum(region * kernel)

    return result


def normalize_image(image):

    image = np.clip(image, 0, 255)

    return image.astype(np.uint8)