import numpy as np


def add_gaussian_noise(image, sigma=25):
    h, w = image.shape

    u1 = np.random.uniform(0.0001, 1, (h, w))
    u2 = np.random.uniform(0.0001, 1, (h, w))

    z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)

    noise = sigma * z

    noisy = image.astype(np.float64) + noise

    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_uniform_noise(image, low=50, high=50):
    h, w = image.shape

    noise = np.random.uniform(-low, high, (h, w))

    noisy = image.astype(np.float64) + noise

    return np.clip(noisy, 0, 255).astype(np.uint8)