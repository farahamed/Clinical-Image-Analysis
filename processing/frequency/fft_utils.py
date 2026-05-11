import numpy as np


def compute_fft(image):

    image = image.astype(np.float32)

    F = np.fft.fft2(image)

    F_shifted = np.fft.fftshift(F)

    return F_shifted


def inverse_fft(F_shifted):

    inverse_shift = np.fft.ifftshift(F_shifted)

    reconstructed = np.fft.ifft2(inverse_shift)

    reconstructed = np.abs(reconstructed)

    reconstructed = np.clip(reconstructed, 0, 255)

    return reconstructed.astype(np.uint8)