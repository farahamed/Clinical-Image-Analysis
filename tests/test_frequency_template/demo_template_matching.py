"""
Demo script to visualize template matching before and after.
Shows both synthetic medical images and real images.

Shows the original image with template highlighted, and result with bounding box.

Run from project root:
    python tests/test_frequency_template/demo_template_matching.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

# Ensure project root is on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from processing.frequency.frequency_template_matching import fourier_cross_correlate, fourier_cross_correlate_normalized


def create_synthetic_medical_image(seed=42):
    """
    Create a synthetic medical-like image with bright and dark regions.
    Returns grayscale uint8 image.
    """
    rng = np.random.default_rng(seed)
    
    # Base background (dark gray, like lung tissue)
    image = rng.integers(30, 80, size=(400, 500), dtype=np.uint8)
    
    # Add bright regions (like bone or dense tissue)
    image[100:180, 150:250] = rng.integers(180, 240, size=(80, 100), dtype=np.uint8)
    image[200:280, 300:400] = rng.integers(170, 230, size=(80, 100), dtype=np.uint8)
    
    # Add some texture
    noise = rng.integers(-20, 20, size=image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return image


def load_real_image(image_path):
    """
    Load a real image from disk. Converts to grayscale if needed.
    Returns grayscale uint8 image or None if not found.
    """
    if not os.path.exists(image_path):
        return None
    
    try:
        pil_img = Image.open(image_path).convert("L")  # Convert to grayscale
        image = np.array(pil_img, dtype=np.uint8)
        return image
    except Exception as e:
        print(f"  [Warning] Could not load {image_path}: {e}")
        return None


def demo_image_comparison(image, image_name, template_row=None, template_col=None, 
                         template_h=50, template_w=70, save_name=None):
    """
    Run BOTH raw and normalized template matching on the same image for comparison.
    
    Parameters
    ----------
    image : np.ndarray
        Grayscale uint8 image
    image_name : str
        Name for display
    template_row, template_col : int
        Where to extract template from. If None, uses center region.
    template_h, template_w : int
        Template height and width
    save_name : str
        Base filename to save visualization (without .png extension)
    """
    print(f"\n{'=' * 70}")
    print(f"  {image_name}")
    print(f"{'=' * 70}")
    
    # Auto-detect template location if not provided
    if template_row is None or template_col is None:
        ih, iw = image.shape
        template_row = max(0, (ih - template_h) // 2)
        template_col = max(0, (iw - template_w) // 2)
        template_row = min(template_row, ih - template_h)
        template_col = min(template_col, iw - template_w)
    
    # Extract template
    print(f"\n[1] Image loaded: shape={image.shape}")
    print(f"[2] Extracting template at ({template_row}, {template_col})")
    template = image[template_row:template_row + template_h, 
                     template_col:template_col + template_w].copy()
    print(f"    Template shape: {template.shape}")
    
    # Run RAW cross-correlation
    print(f"\n[3a] Running RAW Fourier cross-correlation...")
    result_raw, corr_raw, (peak_row_raw, peak_col_raw), (th, tw) = fourier_cross_correlate(
        image, template
    )
    row_err_raw = abs(peak_row_raw - template_row)
    col_err_raw = abs(peak_col_raw - template_col)
    print(f"     Peak found at: ({int(peak_row_raw)}, {int(peak_col_raw)})")
    print(f"     Error: ({int(row_err_raw)}, {int(col_err_raw)}) pixels")
    
    # Run NORMALIZED cross-correlation
    print(f"\n[3b] Running NORMALIZED Fourier cross-correlation...")
    result_norm, corr_norm, (peak_row_norm, peak_col_norm), (th, tw) = fourier_cross_correlate_normalized(
        image, template
    )
    row_err_norm = abs(peak_row_norm - template_row)
    col_err_norm = abs(peak_col_norm - template_col)
    print(f"     Peak found at: ({int(peak_row_norm)}, {int(peak_col_norm)})")
    print(f"     Error: ({int(row_err_norm)}, {int(col_err_norm)}) pixels")
    
    # Create comparison visualization
    print(f"[4] Creating side-by-side comparison visualization...")
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle(f"Template Matching Comparison (Raw vs Normalized) — {image_name}", 
                 fontsize=16, fontweight='bold')
    
    # Row 0: Original and template
    ax = axes[0, 0]
    ax.imshow(image, cmap='gray')
    rect_template = Rectangle((template_col, template_row), template_w, template_h,
                              linewidth=2, edgecolor='cyan', facecolor='none')
    ax.add_patch(rect_template)
    ax.set_title("Original Image\n(Cyan = Template)", fontweight='bold')
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.imshow(template, cmap='gray')
    ax.set_title(f"Template\n({template_h}×{template_w})", fontweight='bold')
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 2]
    ax.text(0.5, 0.5, f"Expected:\n({int(template_row)}, {int(template_col)})",
            ha='center', va='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue'))
    ax.axis('off')
    
    # Row 1: RAW cross-correlation results
    ax = axes[1, 0]
    ax.imshow(result_raw)
    ax.set_title("RAW: Result Image\n(Red box)", fontweight='bold')
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    im = ax.imshow(corr_raw, cmap='hot', origin='upper')
    ax.plot(peak_col_raw, peak_row_raw, 'g+', markersize=15, markeredgewidth=2)
    ax.set_title("RAW: Correlation Map", fontweight='bold')
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Corr", rotation=270, labelpad=10)
    
    ax = axes[1, 2]
    ax.text(0.5, 0.5, f"RAW Result:\n({int(peak_row_raw)}, {int(peak_col_raw)})\n\nError:\n({int(row_err_raw)}, {int(col_err_raw)})",
            ha='center', va='center', fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax.axis('off')
    
    # Row 2: NORMALIZED cross-correlation results
    ax = axes[2, 0]
    ax.imshow(result_norm)
    ax.set_title("NORMALIZED: Result Image\n(Red box)", fontweight='bold')
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.grid(True, alpha=0.3)
    
    ax = axes[2, 1]
    im = ax.imshow(corr_norm, cmap='hot', origin='upper')
    ax.plot(peak_col_norm, peak_row_norm, 'g+', markersize=15, markeredgewidth=2)
    ax.set_title("NORMALIZED: Correlation Map", fontweight='bold')
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("NCC", rotation=270, labelpad=10)
    
    ax = axes[2, 2]
    ax.text(0.5, 0.5, f"NORMALIZED Result:\n({int(peak_row_norm)}, {int(peak_col_norm)})\n\nError:\n({int(row_err_norm)}, {int(col_err_norm)})",
            ha='center', va='center', fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen'))
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save figure
    if save_name:
        output_path = os.path.join(PROJECT_ROOT, f"{save_name}.png")
        print(f"[5] Saving visualization to: {output_path}")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print("    ✓ Saved")
    
    return {
        'raw': {'peak': (peak_row_raw, peak_col_raw), 'error': (row_err_raw, col_err_raw)},
        'norm': {'peak': (peak_row_norm, peak_col_norm), 'error': (row_err_norm, col_err_norm)}
    }


def main():
    print("\n" + "=" * 70)
    print("  Template Matching Demo — Raw vs Normalized Comparison")
    print("=" * 70)
    
    # --- Demo 1: Synthetic medical image ---
    print("\n### DEMO 1: SYNTHETIC MEDICAL IMAGE ###")
    synthetic_image = create_synthetic_medical_image()
    results_synthetic = demo_image_comparison(synthetic_image, "Synthetic Medical Image", 
                                    template_row=100, template_col=150, 
                                    template_h=50, template_w=70,
                                    save_name="template_matching_synthetic_comparison")
    
    # --- Demo 2: Real image from assets ---
    print("\n### DEMO 2: REAL IMAGE FROM ASSETS ###")
    real_image_path = os.path.join(PROJECT_ROOT, "assets", "test.jpeg")
    real_image = load_real_image(real_image_path)
    
    if real_image is not None:
        ih, iw = real_image.shape
        # Select a region with better features/contrast for template
        # Try upper-middle region
        template_h, template_w = min(60, ih // 4), min(70, iw // 4)
        template_row = max(10, ih // 5)  # Avoid very top edge
        template_col = max(10, iw // 5)
        
        results_real = demo_image_comparison(
            real_image, "Real Image from assets/test.jpeg",
            template_row=template_row, template_col=template_col,
            template_h=template_h, template_w=template_w,
            save_name="template_matching_real_comparison"
        )
        
        # Print summary
        print(f"\n{'=' * 70}")
        print("  SUMMARY: Real Image Results")
        print(f"{'=' * 70}")
        print(f"\nRAW Cross-Correlation:")
        print(f"  Found at: ({int(results_real['raw']['peak'][0])}, {int(results_real['raw']['peak'][1])})")
        print(f"  Error:    ({int(results_real['raw']['error'][0])}, {int(results_real['raw']['error'][1])}) pixels")
        print(f"\nNORMALIZED Cross-Correlation:")
        print(f"  Found at: ({int(results_real['norm']['peak'][0])}, {int(results_real['norm']['peak'][1])})")
        print(f"  Error:    ({int(results_real['norm']['error'][0])}, {int(results_real['norm']['error'][1])}) pixels")
    else:
        print(f"\n  [skip] Real image not found at {real_image_path}")
    
    # Display plots
    print("\n[6] Displaying plots (close windows to exit)...")
    plt.show()
    
    print("\n" + "=" * 70)
    print("  Demo complete!")
    print("=" * 70)
    print(f"\n  Visualizations saved to project root:")
    print(f"    - template_matching_synthetic_comparison.png")
    if real_image is not None:
        print(f"    - template_matching_real_comparison.png")
    print()
    print("  KEY INSIGHT:")
    print("  Normalized Cross-Correlation (NCC) is invariant to brightness/contrast")
    print("  changes, making it much more robust for real medical images with")
    print("  varying illumination and texture uniformity.")
    print()


if __name__ == "__main__":
    main()
