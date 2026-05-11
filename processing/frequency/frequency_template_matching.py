import numpy as np

def _to_gray_float(arr: np.ndarray) -> np.ndarray:
    """
    Convert an image array to a 2-D float64 grayscale array.
    Handles both grayscale (H, W) and RGB (H, W, 3) inputs.
    Uses the standard luminance weights — implemented manually.
    """
    if arr.ndim == 2:
        return arr.astype(np.float64)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        # Luminance formula: Y = 0.2989·R + 0.5870·G + 0.1140·B
        return (0.2989 * arr[:, :, 0].astype(np.float64)
                + 0.5870 * arr[:, :, 1].astype(np.float64)
                + 0.1140 * arr[:, :, 2].astype(np.float64))
    raise ValueError(f"Unsupported image shape: {arr.shape}")

def _to_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    """Return a (H, W, 3) uint8 array regardless of whether input is gray or RGB."""
    if arr.ndim == 2:
        channel = np.clip(arr, 0, 255).astype(np.uint8)
        return np.stack([channel, channel, channel], axis=-1)
    return np.clip(arr, 0, 255).astype(np.uint8)

def _draw_rect(image_rgb: np.ndarray,
               r1: int, c1: int, r2: int, c2: int,
               color=(255, 0, 0), thickness: int = 2) -> None:
    """
    Draw a filled-border rectangle on a (H, W, 3) uint8 array **in-place**.
   
    """
    ih, iw = image_rgb.shape[:2]

    r1 = max(r1, 0)
    c1 = max(c1, 0)
    r2 = min(r2, ih - 1)
    c2 = min(c2, iw - 1)

    for t in range(thickness):
        # Horizontal edges (top / bottom)
        if r1 + t < ih:
            image_rgb[r1 + t, c1:c2 + 1] = color
        if r2 - t >= 0:
            image_rgb[r2 - t, c1:c2 + 1] = color
        # Vertical edges (left / right)
        if c1 + t < iw:
            image_rgb[r1:r2 + 1, c1 + t] = color
        if c2 - t >= 0:
            image_rgb[r1:r2 + 1, c2 - t] = color

def fourier_cross_correlate(image: np.ndarray, template: np.ndarray):
    """
    Locate `template` inside `image` using Fourier cross-correlation.

    Returns
    -------
    result_image  : (H, W, 3) uint8 — original image with red bounding box
    norm_corr_map : (H, W) float64  — normalised correlation map in [0, 1]
    peak          : (row, col)      — top-left corner of best match
    (th, tw)      : template height and width
    """

    # ── 1. Grayscale float ────────────────────────────────────────────────────
    gray_image    = _to_gray_float(image)
    gray_template = _to_gray_float(template)

    ih, iw = gray_image.shape
    th, tw = gray_template.shape

    if th > ih or tw > iw:
        raise ValueError(
            f"Template ({th}×{tw}) is larger than the image ({ih}×{iw})."
        )
    if th < 2 or tw < 2:
        raise ValueError("Template must be at least 2×2 pixels.")

    # ── 2. Padded size (linear cross-correlation — no wrap-around) ────────────
    pad_rows = ih + th - 1
    pad_cols = iw + tw - 1

    # ── 3. Zero-pad BOTH to the same padded size ──────────────────────────────
    image_padded = np.zeros((pad_rows, pad_cols), dtype=np.float64)
    image_padded[:ih, :iw] = gray_image          # ← gray_image, not image

    template_padded = np.zeros((pad_rows, pad_cols), dtype=np.float64)
    template_padded[:th, :tw] = gray_template    # ← same padded size as image

    # ── 4. Forward FFTs ───────────────────────────────────────────────────────
    F_image    = np.fft.fft2(image_padded)        # shape: (pad_rows, pad_cols)
    F_template = np.fft.fft2(template_padded)     # shape: (pad_rows, pad_cols) ✓

    # ── 5. Cross-correlation spectrum: C = F · conj(G) ───────────────────────
    corr_spectrum = F_image * np.conj(F_template)
    correlation_map = np.real(np.fft.ifft2(corr_spectrum))

    valid_map = correlation_map[:ih, :iw]

    v_min, v_max = valid_map.min(), valid_map.max()
    if v_max - v_min > 1e-10:
        norm_corr_map = (valid_map - v_min) / (v_max - v_min)
    else:
        norm_corr_map = np.zeros_like(valid_map)

    peak_flat = int(np.argmax(valid_map))
    peak_row, peak_col = np.unravel_index(peak_flat, valid_map.shape)

    result_image = _to_rgb_uint8(image).copy()
    r1, c1 = int(peak_row), int(peak_col)
    r2, c2 = r1 + th - 1, c1 + tw - 1
    _draw_rect(result_image, r1, c1, r2, c2, color=(255, 0, 0), thickness=2)

    return result_image, norm_corr_map, (peak_row, peak_col), (th, tw)


def fourier_cross_correlate_normalized(image: np.ndarray, template: np.ndarray):
    gray_image = _to_gray_float(image)
    gray_template = _to_gray_float(template)

    ih, iw = gray_image.shape
    th, tw = gray_template.shape
    N = float(th * tw)

    if th > ih or tw > iw:
        raise ValueError(f"Template ({th}×{tw}) is larger than the image ({ih}×{iw}).")
    if th < 2 or tw < 2:
        raise ValueError("Template must be at least 2×2 pixels.")

    # ── FIX 2: subtract global mean BEFORE squaring ─────────────────────────
    gray_image -= gray_image.mean()

    # ── Normalize template (zero-mean, unit std) ─────────────────────────────
    t_mean = gray_template.mean()
    t_std = gray_template.std()
    if t_std < 1e-6:
        result_image = _to_rgb_uint8(image).copy()
        _draw_rect(result_image, 0, 0, th - 1, tw - 1, color=(255, 0, 0), thickness=2)
        return result_image, np.zeros((ih, iw)), (0, 0), (th, tw)
    norm_template = (gray_template - t_mean) / t_std

    pad_rows = ih + th - 1
    pad_cols = iw + tw - 1

    img_padded = np.zeros((pad_rows, pad_cols), dtype=np.float64)
    img_padded[:ih, :iw] = gray_image

    img_sq_padded = np.zeros((pad_rows, pad_cols), dtype=np.float64)
    img_sq_padded[:ih, :iw] = gray_image ** 2

    tmpl_padded = np.zeros((pad_rows, pad_cols), dtype=np.float64)
    tmpl_padded[:th, :tw] = norm_template

    ones_padded = np.zeros((pad_rows, pad_cols), dtype=np.float64)
    ones_padded[:th, :tw] = 1.0

    F_img = np.fft.fft2(img_padded)
    F_img_sq = np.fft.fft2(img_sq_padded)
    F_tmpl = np.fft.fft2(tmpl_padded)
    F_ones = np.fft.fft2(ones_padded)

    numerator_full = np.real(np.fft.ifft2(F_img * np.conj(F_tmpl)))
    local_sum_full = np.real(np.fft.ifft2(F_img * np.conj(F_ones)))
    local_sum_sq_full = np.real(np.fft.ifft2(F_img_sq * np.conj(F_ones)))

    out_h = ih - th + 1
    out_w = iw - tw + 1

    # ── FIX 1: valid region starts at index 0, NOT th-1 ──────────────────────
    numerator = numerator_full[:out_h, :out_w]
    local_sum = local_sum_full[:out_h, :out_w]
    local_sum_sq = local_sum_sq_full[:out_h, :out_w]

    local_mean = local_sum / N
    local_mean_sq = local_sum_sq / N
    local_var = np.clip(local_mean_sq - local_mean ** 2, 0.0, None)
    local_std = np.sqrt(local_var)

    denom = N * local_std
    safe = denom > 1e-12
    ncc_valid = np.zeros((out_h, out_w), dtype=np.float64)
    ncc_valid[safe] = numerator[safe] / denom[safe]
    ncc_valid = np.clip(ncc_valid, -1.0, 1.0)

    # ── Memory optimization: avoid redundant full-size array ──────────────────
    norm_corr_map = np.zeros((ih, iw), dtype=np.float64)
    norm_corr_map[:out_h, :out_w] = (ncc_valid + 1.0) / 2.0

    peak_flat = int(np.argmax(ncc_valid))
    peak_row, peak_col = np.unravel_index(peak_flat, (out_h, out_w))

    result_image = _to_rgb_uint8(image).copy()
    r1, c1 = int(peak_row), int(peak_col)
    r2, c2 = r1 + th - 1, c1 + tw - 1
    _draw_rect(result_image, r1, c1, r2, c2, color=(255, 0, 0), thickness=2)

    return result_image, norm_corr_map, (peak_row, peak_col), (th, tw)