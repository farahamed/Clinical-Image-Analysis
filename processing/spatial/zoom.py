import numpy as np


def zoom_nearest(img, scale):
    h, w = img.shape
    new_h = int(h * scale)
    new_w = int(w * scale)

    row_indices = (np.arange(new_h) / scale).astype(int)
    col_indices = (np.arange(new_w) / scale).astype(int)

    row_indices = np.clip(row_indices, 0, h - 1)
    col_indices = np.clip(col_indices, 0, w - 1)

    return img[np.ix_(row_indices, col_indices)]


def zoom_bilinear(img, scale):
    h, w = img.shape
    new_h = int(h * scale)
    new_w = int(w * scale)

    new_cols = np.arange(new_w) / scale
    new_rows = np.arange(new_h) / scale

    col_grid, row_grid = np.meshgrid(new_cols, new_rows)

    x1 = np.floor(row_grid).astype(int)
    y1 = np.floor(col_grid).astype(int)
    x2 = np.clip(x1 + 1, 0, h - 1)
    y2 = np.clip(y1 + 1, 0, w - 1)
    x1 = np.clip(x1, 0, h - 1)
    y1 = np.clip(y1, 0, w - 1)

    dx = row_grid - np.floor(row_grid)
    dy = col_grid - np.floor(col_grid)

    val = (
        img[x1, y1] * (1 - dx) * (1 - dy) +
        img[x2, y1] * dx       * (1 - dy) +
        img[x1, y2] * (1 - dx) * dy       +
        img[x2, y2] * dx       * dy
    )

    return np.clip(val, 0, 255).astype(np.uint8)


def zoom_image(image_array, scale_factor, method="bilinear"):
    if image_array is None:
        return None
    if scale_factor <= 0:
        return image_array
    if method == "nearest":
        return zoom_nearest(image_array, scale_factor)
    else:
        return zoom_bilinear(image_array, scale_factor)