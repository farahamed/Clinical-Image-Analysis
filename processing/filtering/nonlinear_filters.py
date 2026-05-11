import numpy as np


def median_filter(image, size):

    image = image.astype(np.float32)

    h, w = image.shape

    k = size // 2

    padded = np.pad(
        image,
        ((k, k), (k, k)),
        mode='edge'
    )

    result = np.zeros((h, w), dtype=np.float32)

    median_index = (size * size) // 2

    for i in range(h):

        rows = padded[i:i + size]

        for j in range(w):

            window = rows[:, j:j + size].ravel()

            result[i, j] = np.partition(
                window,
                median_index
            )[median_index]

    return result