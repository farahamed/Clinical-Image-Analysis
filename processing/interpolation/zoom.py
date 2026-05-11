import numpy as np


def zoom_nearest(img, scale):
    #if RGB, process every channel separately, recombine channels
    if img.ndim == 3:
        channels=[zoom_nearest(img[:,:,c], scale) for c in range(img.shape[2])]
        return np.stack(channels, axis=2)
    h, w = img.shape

    new_h = int(h * scale)
    new_w = int(w * scale)
    #inverse mapping, new -> old
    row_indices = (np.arange(new_h) / scale).astype(int)
    col_indices = (np.arange(new_w) / scale).astype(int)

    #prevents out of bounds
    row_indices = np.clip(row_indices, 0, h - 1)
    col_indices = np.clip(col_indices, 0, w - 1)
    
    #builds output image, ix is equivalent to nested loops, but faster, creates grid of indices for new image, maps to old image
    return img[np.ix_(row_indices, col_indices)]


def zoom_bilinear(img, scale):
    if img.ndim == 3:
        channels=[zoom_bilinear(img[:,:,c], scale) for c in range(img.shape[2])]
        return np.stack(channels, axis=2)
    
    h, w = img.shape
    #define size of output image
    new_h = int(h * scale)
    new_w = int(w * scale)
    #Mapping indices,[0, 1, 2, ..., new_h-1]
    new_cols = np.arange(new_w) / scale
    new_rows = np.arange(new_h) / scale
    #Create full coordinate grid, mapping coordinates
    col_grid, row_grid = np.meshgrid(new_cols, new_rows)
    #Neighbor pixels
    #Top left pixels
    x1 = np.floor(row_grid).astype(int)
    y1 = np.floor(col_grid).astype(int)

    #Bottom right pixels
    x2 = np.clip(x1 + 1, 0, h - 1)
    y2 = np.clip(y1 + 1, 0, w - 1)

    #clip values
    x1 = np.clip(x1, 0, h - 1)
    y1 = np.clip(y1, 0, w - 1)

    #fractional distances
    dx = row_grid - np.floor(row_grid)
    dy = col_grid - np.floor(col_grid)
    #interpolation formula, weighted average of 4 neighbors
    val = (
        img[x1, y1] * (1 - dx) * (1 - dy) +
        img[x2, y1] * dx       * (1 - dy) +
        img[x1, y2] * (1 - dx) * dy       +
        img[x2, y2] * dx       * dy
    )
    #clamp values to valid range and convert to uint8
    return np.clip(val, 0, 255).astype(np.uint8)

def bilinear_interpolation_pixel(img,x_src,y_src):
    h, w = img.shape[:2]
    #find the 4 surrounding pixels (neigbhoring)
    x0, y0 = int(x_src), int(y_src)
    x1, y1 = x0 + 1, y0 + 1
    #bondary handling
    if x0 <0 or y0 <0 or x1 >= w or y1 >= h:
        return 0
    #fractional part
    a= x_src - x0
    b= y_src - y0   
    #bilinear interpolation formula, weighted average of 4 neighbors, weights based on distance from source pixel
    return(
        (1 - a) * (1 - b) * img[y0, x0] +
        a * (1 - b) * img[y0, x1] +
        (1 - a) * b * img[y1, x0] +
        a * b * img[y1, x1]
    )


def zoom_image(image_array, scale_factor, method="bilinear"):
    if image_array is None:
        return None
    #invalid scale handling
    if scale_factor <= 0:
        return image_array
    #select method
    if method == "nearest":
        return zoom_nearest(image_array, scale_factor)
    else:
        return zoom_bilinear(image_array, scale_factor)
    
 