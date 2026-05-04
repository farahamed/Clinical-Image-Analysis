import numpy as np
from .convolution import convolve2d


def sobel_kernels():
    gx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)

    gy = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float64)

    return gx, gy

def apply_edge_detection(image):
     gx, gy = sobel_kernels()
     grad_x = convolve2d(image, gx)
     grad_y = convolve2d(image, gy)

     magnitude = np.sqrt(grad_x**2 + grad_y**2)

     return grad_x, grad_y, magnitude
    