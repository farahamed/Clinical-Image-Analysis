# processing/roi/roi_stats.py

import numpy as np

def compute_roi_histogram(roi: np.ndarray, num_bins: int = 256):
    """
    Compute histogram of pixel values from scratch.
    No np.histogram allowed.
    Returns: hist (array of counts), bin_edges (0–255)
    """
    flat = roi.flatten().astype(np.int32)
    hist = np.zeros(num_bins, dtype=np.int64)

    for pixel in flat:
        idx = max(0, min(int(pixel), num_bins - 1))
        hist[idx] += 1

    bin_edges = np.arange(num_bins + 1)
    return hist, bin_edges


def compute_roi_mean(roi: np.ndarray) -> float:
    """
    Mean from scratch.
    Formula: sum(pixels) / count
    """
    flat  = roi.flatten().astype(np.float64)
    total = 0.0
    for px in flat:
        total += px
    return total / len(flat)


def compute_roi_variance(roi: np.ndarray) -> float:
    """
    Variance from scratch.
    Formula: sum((xi - mean)^2) / N
    """
    flat     = roi.flatten().astype(np.float64)
    mean     = compute_roi_mean(roi)
    sq_sum   = 0.0
    for px in flat:
        sq_sum += (px - mean) ** 2
    return sq_sum / len(flat)