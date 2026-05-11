import numpy as np


def magnitude_spectrum(F_shifted):

    magnitude = np.abs(F_shifted)

    spectrum = np.log1p(magnitude)

    spectrum = spectrum / spectrum.max()

    spectrum = (spectrum * 255).astype(np.uint8)

    return spectrum