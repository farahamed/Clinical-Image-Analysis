import numpy as np


def distance_matrix(shape, center_u, center_v):

    rows, cols = shape

    U, V = np.meshgrid(
        np.arange(cols),
        np.arange(rows)
    )

    D = np.sqrt(
        (U - center_v) ** 2 +
        (V - center_u) ** 2
    )

    return D


def ideal_notch(shape, points, radius):

    rows, cols = shape

    mask = np.ones(shape, dtype=np.float32)

    for (u, v) in points:

        D1 = distance_matrix(shape, u, v)

        mirror_u = rows - u
        mirror_v = cols - v

        D2 = distance_matrix(
            shape,
            mirror_u,
            mirror_v
        )

        mask[D1 <= radius] = 0
        mask[D2 <= radius] = 0

    return mask


def butterworth_notch(
    shape,
    points,
    radius,
    order=2
):

    mask = np.ones(shape, dtype=np.float32)

    epsilon = 1e-6

    rows, cols = shape

    for (u, v) in points:

        mirror_u = rows - u
        mirror_v = cols - v

        D1 = distance_matrix(shape, u, v)
        D2 = distance_matrix(shape, mirror_u, mirror_v)

        notch1 = 1 / (
            1 + (radius / (D1 + epsilon)) ** (2 * order)
        )

        notch2 = 1 / (
            1 + (radius / (D2 + epsilon)) ** (2 * order)
        )

        mask *= notch1 * notch2

    return mask


def gaussian_notch(shape, points, radius):

    mask = np.ones(shape, dtype=np.float32)

    rows, cols = shape

    for (u, v) in points:

        mirror_u = rows - u
        mirror_v = cols - v

        D1 = distance_matrix(shape, u, v)
        D2 = distance_matrix(shape, mirror_u, mirror_v)

        notch1 = 1 - np.exp(
            -(D1 ** 2) / (2 * radius ** 2)
        )

        notch2 = 1 - np.exp(
            -(D2 ** 2) / (2 * radius ** 2)
        )

        mask *= notch1 * notch2

    return mask