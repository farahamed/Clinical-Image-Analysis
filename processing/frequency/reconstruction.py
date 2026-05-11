import numpy as np


def reconstruct_image(filtered_fft):

    inverse_shift = np.fft.ifftshift(filtered_fft)

    reconstructed = np.fft.ifft2(inverse_shift)

    reconstructed = np.abs(reconstructed)

    reconstructed = np.clip(reconstructed, 0, 255)

    return reconstructed.astype(np.uint8)