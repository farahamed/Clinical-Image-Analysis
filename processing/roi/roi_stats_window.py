# processing/roi/roi_stats_window.py

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from processing.roi.roi_stats import (
    compute_roi_histogram,
    compute_roi_mean,
    compute_roi_variance
)


def show_roi_statistics(roi: np.ndarray, roi_label: str = "Selected ROI"):
    """
    Compute and display histogram + mean + variance for the given ROI.
    Opens a matplotlib popup window.

    Args:
        roi       : 2D numpy array of pixel values (the isolated region)
        roi_label : string shown in the title (e.g. "ROI 120×85 px")
    """
    if roi is None or roi.size == 0:
        return

    # Convert to grayscale if RGB
    if roi.ndim == 3:
        roi = (0.299 * roi[:, :, 0] +
               0.587 * roi[:, :, 1] +
               0.114 * roi[:, :, 2]).astype(np.uint8)

    # ── Compute stats from scratch ──────────────────────────────
    hist, bin_edges = compute_roi_histogram(roi)
    mean            = compute_roi_mean(roi)
    variance        = compute_roi_variance(roi)
    std_dev         = variance ** 0.5

    # ── Plot ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"ROI Statistics — {roi_label}",
        fontsize=14, fontweight="bold", color="#1A3A6B"
    )
    fig.patch.set_facecolor("#1a1a2e")

    # ── Left: ROI image preview ───────────────────────────────────
    ax_img = axes[0]
    ax_img.imshow(roi, cmap="gray", vmin=0, vmax=255)
    ax_img.set_title("ROI Preview", color="white", fontsize=12)
    ax_img.axis("off")
    ax_img.set_facecolor("#2b2b2b")

    # ── Right: Histogram ──────────────────────────────────────────
    ax_hist = axes[1]
    ax_hist.set_facecolor("#2b2b2b")

    # Bar chart of histogram
    ax_hist.bar(
        bin_edges[:-1], hist,
        width=1, color="#2E75B6", alpha=0.85, label="Pixel count"
    )

    # Mean line
    ax_hist.axvline(
        mean, color="#e94560", linewidth=2,
        linestyle="--", label=f"Mean = {mean:.2f}"
    )

    # ±1 std dev shaded region
    ax_hist.axvspan(
        max(0, mean - std_dev), min(255, mean + std_dev),
        alpha=0.15, color="#e94560", label=f"±1 Std Dev = {std_dev:.2f}"
    )

    ax_hist.set_xlabel("Pixel Intensity (0–255)", color="white", fontsize=11)
    ax_hist.set_ylabel("Pixel Count",             color="white", fontsize=11)
    ax_hist.set_xlim(0, 255)
    ax_hist.tick_params(colors="white")
    for spine in ax_hist.spines.values():
        spine.set_edgecolor("#555555")

    ax_hist.set_title("Local Histogram", color="white", fontsize=12)
    ax_hist.legend(facecolor="#2b2b2b", labelcolor="white", fontsize=10)

    # ── Stats text box ────────────────────────────────────────────
    stats_text = (
        f"Pixels:    {roi.size:,}\n"
        f"Mean:      {mean:.4f}\n"
        f"Variance:  {variance:.4f}\n"
        f"Std Dev:   {std_dev:.4f}\n"
        f"Min:       {int(roi.min())}\n"
        f"Max:       {int(roi.max())}"
    )
    ax_hist.text(
        0.98, 0.97, stats_text,
        transform=ax_hist.transAxes,
        fontsize=10, verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#0f3460",
                  edgecolor="#2E75B6", alpha=0.9),
        color="white", fontfamily="monospace"
    )

    plt.tight_layout()
    plt.show()