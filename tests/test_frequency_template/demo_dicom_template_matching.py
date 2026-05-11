"""
Template Matching Demo on Real DICOM CT Image

Uses the actual DICOM CT image from the geometric transformations test.
Demonstrates template matching on real medical imaging data.

Run from project root:
    python tests/test_frequency_template/demo_dicom_template_matching.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pydicom

# Ensure project root is on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from processing.frequency.frequency_template_matching import fourier_cross_correlate_normalized
from image_io.image_loader import load_dicom_image


def find_dicom_file():
    """
    Search for DICOM files in the Kaggle cache location.
    Returns the path to the first DICOM file found.
    """
    dicom_folder = r"C:\Users\DELL\.cache\kagglehub\datasets\kmader\siim-medical-images\versions\6"
    
    if not os.path.exists(dicom_folder):
        return None
    
    for root, dirs, files in os.walk(dicom_folder):
        for file in files:
            if file.endswith(".dcm"):
                return os.path.join(root, file)
    
    return None


def main():
    print("\n" + "=" * 70)
    print("  Template Matching on Real DICOM CT Image")
    print("=" * 70)
    
    # Find and load DICOM image
    print("\n[1] Searching for DICOM image...")
    dicom_path = find_dicom_file()
    
    if dicom_path is None:
        print("  [ERROR] DICOM file not found in Kaggle cache.")
        print("  Expected location: C:\\Users\\DELL\\.cache\\kagglehub\\datasets\\kmader\\siim-medical-images\\versions\\6")
        return
    
    print(f"  Found: {dicom_path}")
    
    print("\n[2] Loading DICOM image...")
    try:
        gray_image, dicom_data = load_dicom_image(dicom_path)
        print(f"  Image shape: {gray_image.shape}")
        print(f"  Pixel range: [{gray_image.min()}, {gray_image.max()}]")
    except Exception as e:
        print(f"  [ERROR] Failed to load DICOM: {e}")
        return
    
    ih, iw = gray_image.shape
    
    # For real CT image, use a smaller region to extract template
    # This avoids edge artifacts and makes template more distinctive
    print("\n[3] Extracting template from CT image...")
    template_h, template_w = 40, 50
    # Choose a region in the middle that likely has distinctive features
    template_row = max(20, (ih - template_h) // 2)
    template_col = max(20, (iw - template_w) // 2)
    
    template = gray_image[template_row:template_row + template_h,
                         template_col:template_col + template_w].copy()
    
    print(f"  Template extracted at ({template_row}, {template_col})")
    print(f"  Template shape: {template.shape}")
    print(f"  Template range: [{template.min()}, {template.max()}]")
    
    # Run template matching
    print("\n[4] Running Fourier normalized cross-correlation...")
    result_image, norm_corr_map, (peak_row, peak_col), (th, tw) = fourier_cross_correlate_normalized(
        gray_image, template
    )
    
    row_err = abs(peak_row - template_row)
    col_err = abs(peak_col - template_col)
    
    print(f"  Peak found at: ({int(peak_row)}, {int(peak_col)})")
    print(f"  Expected at:  ({template_row}, {template_col})")
    print(f"  Error: ({int(row_err)}, {int(col_err)}) pixels")
    
    # Determine if match was successful
    if row_err <= 3 and col_err <= 3:
        print("  ✓ MATCH SUCCESSFUL (error ≤ 3 pixels)")
    elif row_err <= 10 and col_err <= 10:
        print("  ~ WEAK MATCH (error ≤ 10 pixels)")
    else:
        print("  ✗ MATCH FAILED (high error)")
    
    # Create visualization
    print("\n[5] Creating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Template Matching on Real CT DICOM Image", 
                 fontsize=16, fontweight='bold')
    
    # --- Original image with template region ---
    ax = axes[0, 0]
    ax.imshow(gray_image, cmap='gray')
    rect_template = Rectangle((template_col, template_row), template_w, template_h,
                              linewidth=2, edgecolor='cyan', facecolor='none',
                              label='Template region')
    ax.add_patch(rect_template)
    ax.set_title("Original CT Image (Cyan = Template)", fontweight='bold', fontsize=11)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # --- Extracted template ---
    ax = axes[0, 1]
    ax.imshow(template, cmap='gray')
    ax.set_title(f"Extracted Template ({template_h}×{template_w})", fontweight='bold', fontsize=11)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.grid(True, alpha=0.3)
    
    # --- Result with bounding box ---
    ax = axes[1, 0]
    # Convert grayscale to RGB for visualization
    img_rgb = np.stack([gray_image, gray_image, gray_image], axis=-1)
    ax.imshow(img_rgb)
    rect_match = Rectangle((peak_col, peak_row), template_w, template_h,
                           linewidth=2, edgecolor='red', facecolor='none',
                           label=f'Match at ({int(peak_row)}, {int(peak_col)})')
    ax.add_patch(rect_match)
    ax.set_title("Result: Detected Match (Red Box)", fontweight='bold', fontsize=11)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # --- Correlation map heatmap ---
    ax = axes[1, 1]
    im = ax.imshow(norm_corr_map, cmap='hot', origin='upper')
    ax.plot(peak_col, peak_row, 'g+', markersize=15, markeredgewidth=2,
            label=f'Peak at ({int(peak_row)}, {int(peak_col)})')
    ax.set_title("Correlation Map (Hot colormap)", fontweight='bold', fontsize=11)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.legend(loc='upper right')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Normalized Correlation", rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(PROJECT_ROOT, "template_matching_dicom_ct.png")
    print(f"[6] Saving visualization to: {output_path}")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print("    ✓ Saved")
    
    # Display
    print("\n[7] Displaying plot (close window to exit)...")
    plt.show()
    
    print("\n" + "=" * 70)
    print("  Demo complete!")
    print("=" * 70)
    print(f"\n  Results Summary:")
    print(f"  ──────────────")
    print(f"  DICOM CT Image: {os.path.basename(dicom_path)}")
    print(f"  Image dimensions: {gray_image.shape}")
    print(f"  Template size: {template_h}×{template_w}")
    print(f"  Detection accuracy: Error = ({int(row_err)}, {int(col_err)}) pixels")
    print(f"\n  Visualization saved to: template_matching_dicom_ct.png\n")


if __name__ == "__main__":
    main()
