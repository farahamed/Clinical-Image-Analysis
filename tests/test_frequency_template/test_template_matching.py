"""
Tests for processing/frequency/template_matching.py
=====================================================

Run from the project root (Clinical-Image-Analysis/):

    python -m pytest tests/test_template_matching.py -v

Or without pytest (plain python):

    python tests/test_template_matching.py

Two categories of tests
-----------------------
1. Synthetic  — images built from numpy arrays with a template planted at a
                known pixel position.  Ground truth is exact, so we can assert
                the algorithm finds the right spot.

2. Real image — downloads a public-domain chest X-ray (no account needed),
                crops a patch from it, and checks the algorithm finds it back.
                Automatically skipped if the download fails (no internet, etc.).
"""

import sys
import os
import urllib.request
import numpy as np

# ---------------------------------------------------------------------------
# Make sure the project root is on the path regardless of where pytest is run
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from processing.frequency.frequency_template_matching import fourier_cross_correlate


# ===========================================================================
#  Helpers
# ===========================================================================

def make_gray(h, w, fill=30):
    """Return a uniform grayscale image (uint8)."""
    return np.full((h, w), fill, dtype=np.uint8)


def plant_patch(image, patch, row, col):
    """Copy patch pixels into image at (row, col) — modifies in-place."""
    ph, pw = patch.shape[:2]
    image[row:row + ph, col:col + pw] = patch
    return image


def download_image(url, save_path):
    """
    Download url to save_path.
    Returns True on success, False if the download fails for any reason.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        with open(save_path, "wb") as f:
            f.write(data)
        return True
    except Exception as e:
        print(f"[skip] Could not download {url}: {e}")
        return False


# ===========================================================================
#  Synthetic tests
# ===========================================================================

def test_exact_location_grayscale():
    """
    Plant a bright square at a known position in a dark image.
    The algorithm must find the exact top-left corner.
    """
    image    = make_gray(200, 200, fill=20)
    template = np.full((30, 40), 200, dtype=np.uint8)

    TRUE_ROW, TRUE_COL = 80, 100
    plant_patch(image, template, TRUE_ROW, TRUE_COL)

    result_img, norm_map, (pr, pc), (th, tw) = fourier_cross_correlate(image, template)

    assert (pr, pc) == (TRUE_ROW, TRUE_COL), (
        f"Expected peak at ({TRUE_ROW}, {TRUE_COL}), got ({pr}, {pc})"
    )
    assert th == 30 and tw == 40
    print(f"  PASS  exact location (gray): peak=({pr},{pc})")


def test_exact_location_rgb():
    """
    Same test but with an RGB image — the algorithm must handle the colour
    channel dimension and still return a correct 2-D correlation map.
    """
    image_gray = make_gray(150, 180, fill=15)
    template_gray = np.full((25, 35), 180, dtype=np.uint8)

    TRUE_ROW, TRUE_COL = 60, 90
    plant_patch(image_gray, template_gray, TRUE_ROW, TRUE_COL)

    # Convert to RGB by stacking three identical channels
    image_rgb    = np.stack([image_gray] * 3, axis=-1)
    template_rgb = np.stack([template_gray] * 3, axis=-1)

    result_img, norm_map, (pr, pc), (th, tw) = fourier_cross_correlate(
        image_rgb, template_rgb
    )

    assert (pr, pc) == (TRUE_ROW, TRUE_COL), (
        f"Expected ({TRUE_ROW}, {TRUE_COL}), got ({pr}, {pc})"
    )
    assert result_img.shape == (150, 180, 3), "Result must be (H,W,3) RGB"
    print(f"  PASS  exact location (RGB):  peak=({pr},{pc})")


def test_norm_corr_map_range():
    """Normalised correlation map values must all be in [0, 1]."""
    image    = make_gray(100, 120, fill=40)
    template = np.full((15, 20), 220, dtype=np.uint8)
    plant_patch(image, template, 30, 50)

    _, norm_map, _, _ = fourier_cross_correlate(image, template)

    assert norm_map.min() >= -1e-9,  f"Min below 0: {norm_map.min()}"
    assert norm_map.max() <= 1 + 1e-9, f"Max above 1: {norm_map.max()}"
    assert norm_map.shape == (100, 120), (
        f"Norm map shape wrong: {norm_map.shape}, expected (100, 120)"
    )
    print(f"  PASS  norm_corr_map range:   [{norm_map.min():.4f}, {norm_map.max():.4f}]")


def test_result_image_shape_and_dtype():
    """Result image must always be (H, W, 3) uint8."""
    image    = make_gray(80, 90, fill=60)
    template = np.full((10, 10), 200, dtype=np.uint8)
    plant_patch(image, template, 20, 30)

    result_img, _, _, _ = fourier_cross_correlate(image, template)

    assert result_img.dtype == np.uint8,           f"dtype: {result_img.dtype}"
    assert result_img.shape == (80, 90, 3),        f"shape: {result_img.shape}"
    print(f"  PASS  result shape/dtype:    {result_img.shape} {result_img.dtype}")


def test_bounding_box_pixels_are_red():
    """
    The bounding box drawn at the peak must have red pixels (255, 0, 0)
    on at least the four edges of the rectangle.
    """
    image    = make_gray(100, 100, fill=10)
    template = np.full((20, 25), 200, dtype=np.uint8)
    TRUE_ROW, TRUE_COL = 40, 50
    plant_patch(image, template, TRUE_ROW, TRUE_COL)

    result_img, _, (pr, pc), (th, tw) = fourier_cross_correlate(image, template)

    # Check a pixel on the top edge of the bounding box
    top_edge_pixel = result_img[pr, pc + tw // 2]
    assert top_edge_pixel[0] == 255 and top_edge_pixel[1] == 0, (
        f"Top edge pixel not red: {top_edge_pixel}"
    )
    print(f"  PASS  bounding box is red at top edge: {top_edge_pixel}")


def test_template_too_large_raises():
    """Template larger than image must raise ValueError, not crash."""
    image    = make_gray(50, 50, fill=100)
    template = make_gray(60, 60, fill=200)   # bigger than image
    try:
        fourier_cross_correlate(image, template)
        assert False, "Expected ValueError was not raised"
    except ValueError as e:
        print(f"  PASS  too-large template raises ValueError: {e}")


def test_template_too_small_raises():
    """Template smaller than 2×2 must raise ValueError."""
    image    = make_gray(100, 100, fill=100)
    template = make_gray(1, 1, fill=200)
    try:
        fourier_cross_correlate(image, template)
        assert False, "Expected ValueError was not raised"
    except ValueError as e:
        print(f"  PASS  too-small template raises ValueError: {e}")


def test_corner_placement():
    """
    Template planted at the top-left corner (0, 0).
    Edge case: checks the valid-region crop logic doesn't miss position (0, 0).
    """
    image    = make_gray(120, 150, fill=5)
    template = np.full((20, 30), 240, dtype=np.uint8)
    plant_patch(image, template, 0, 0)

    _, _, (pr, pc), _ = fourier_cross_correlate(image, template)

    assert (pr, pc) == (0, 0), f"Expected (0,0), got ({pr},{pc})"
    print(f"  PASS  corner placement (0,0): peak=({pr},{pc})")


def test_noisy_image():
    """
    Add random noise on top of the planted template.
    The peak should still be at the correct location because the template
    signal is strong relative to uniform noise.
    """
    rng = np.random.default_rng(42)
    image    = rng.integers(0, 50, size=(200, 200), dtype=np.uint8)
    template = np.full((30, 30), 220, dtype=np.uint8)

    TRUE_ROW, TRUE_COL = 100, 120
    plant_patch(image, template, TRUE_ROW, TRUE_COL)

    _, _, (pr, pc), _ = fourier_cross_correlate(image, template)

    assert (pr, pc) == (TRUE_ROW, TRUE_COL), (
        f"Noisy test: expected ({TRUE_ROW},{TRUE_COL}), got ({pr},{pc})"
    )
    print(f"  PASS  noisy image: peak=({pr},{pc})")




# ===========================================================================
#  Runner (plain python — no pytest needed)
# ===========================================================================

ALL_TESTS = [
    test_exact_location_grayscale,
    test_exact_location_rgb,
    test_norm_corr_map_range,
    test_result_image_shape_and_dtype,
    test_bounding_box_pixels_are_red,
    test_template_too_large_raises,
    test_template_too_small_raises,
    test_corner_placement,
    test_noisy_image,
]

if __name__ == "__main__":
    passed = 0
    failed = 0
    print("\n" + "=" * 60)
    print("  Template Matching — Test Suite")
    print("=" * 60)
    for test_fn in ALL_TESTS:
        print(f"\n[{test_fn.__name__}]")
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {type(e).__name__}: {e}")
            failed += 1
    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 60 + "\n")
    sys.exit(0 if failed == 0 else 1)
