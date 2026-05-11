from .notch_mask import (
    ideal_notch,
    butterworth_notch,
    gaussian_notch
)


def apply_notch_filter(
    F_shifted,
    points,
    filter_type='ideal',
    radius=10,
    order=2
):

    shape = F_shifted.shape

    if filter_type == 'ideal':

        mask = ideal_notch(
            shape,
            points,
            radius
        )

    elif filter_type == 'butterworth':

        mask = butterworth_notch(
            shape,
            points,
            radius,
            order
        )

    elif filter_type == 'gaussian':

        mask = gaussian_notch(
            shape,
            points,
            radius
        )

    else:

        raise ValueError("Invalid filter type")

    filtered = F_shifted * mask

    return filtered, mask