import numpy as np
def _to_gray_float(arr: np.ndarray) -> np.ndarray:
    """
    Convert an image array to a 2-D float64 grayscale array.
    Handles both grayscale (H, W) and RGB (H, W, 3) inputs.
    Uses the standard luminance weights — implemented manually, no cv2/PIL.
    """
    if arr.ndim == 2:
        return arr.astype(np.float64)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        # Luminance formula: Y = 0.2989·R + 0.5870·G + 0.1140·B
        return (0.2989 * arr[:, :, 0].astype(np.float64)
                + 0.5870 * arr[:, :, 1].astype(np.float64)
                + 0.1140 * arr[:, :, 2].astype(np.float64))
    raise ValueError(f"Unsupported image shape: {arr.shape}")

