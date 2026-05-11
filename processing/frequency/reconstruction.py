from .fft_utils import inverse_fft


def reconstruct_image(filtered_spectrum):

    return inverse_fft(filtered_spectrum)