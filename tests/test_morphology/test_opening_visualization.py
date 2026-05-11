import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import sys
import os

# Add the project root to the Python path to import from processing module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the functions from your existing module
from processing.morphology.binary_morphology import opening, erode, dilate, create_structuring_element, convert_to_grayscale, ensure_binary


def visualize_opening(original, eroded, opened, structuring_element, title="Opening Result"):
    """Visualize the full opening process: Original → Erosion → Dilation = Opening"""
    
    fig = plt.figure(figsize=(18, 5))
    
    # Original image
    ax1 = plt.subplot(1, 4, 1)
    ax1.imshow(original, cmap='gray', vmin=0, vmax=255)
    white_orig = np.sum(original == 255)
    ax1.set_title(f'1) Original\nWhite: {white_orig} px\nShape: {original.shape}', fontsize=10, fontweight='bold')
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Add pixel values for small images
    if original.size <= 100:
        for i in range(original.shape[0] + 1):
            ax1.axhline(y=i - 0.5, color='blue', linewidth=0.5, alpha=0.3)
        for i in range(original.shape[0]):
            for j in range(original.shape[1]):
                if original[i, j] == 255:
                    ax1.text(j, i, '■', ha='center', va='center', color='red', fontsize=8)
    
    # Structuring Element
    ax2 = plt.subplot(1, 4, 2)
    ax2.imshow(structuring_element, cmap='gray', vmin=0, vmax=1)
    ax2.set_title(f'Structuring Element\nShape: {structuring_element.shape}\nActive: {np.sum(structuring_element)} px', 
                 fontsize=10, fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yticks([])
    for i in range(structuring_element.shape[0] + 1):
        ax2.axhline(y=i - 0.5, color='green', linewidth=1, alpha=0.5)
        ax2.axvline(x=i - 0.5, color='green', linewidth=1, alpha=0.5)
    
    # Erosion result (intermediate step)
    ax3 = plt.subplot(1, 4, 3)
    ax3.imshow(eroded, cmap='gray', vmin=0, vmax=255)
    white_eroded = np.sum(eroded == 255)
    removed = white_orig - white_eroded
    ax3.set_title(f'2) Erosion\nWhite: {white_eroded} px\nRemoved: {removed} px', fontsize=10, fontweight='bold')
    ax3.set_xticks([])
    ax3.set_yticks([])
    
    # Highlight removed pixels
    if original.size <= 400:
        removed_mask = (original == 255) & (eroded == 0)
        for i in range(eroded.shape[0]):
            for j in range(eroded.shape[1]):
                if removed_mask[i, j]:
                    rect = Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                   linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
                    ax3.add_patch(rect)
    
    if eroded.size <= 100:
        for i in range(eroded.shape[0] + 1):
            ax3.axhline(y=i - 0.5, color='blue', linewidth=0.5, alpha=0.3)
    
    # Opening result (final)
    ax4 = plt.subplot(1, 4, 4)
    ax4.imshow(opened, cmap='gray', vmin=0, vmax=255)
    white_opened = np.sum(opened == 255)
    restored = white_opened - white_eroded
    ax4.set_title(f'3) Dilation = Opening\nWhite: {white_opened} px\nRestored: {restored} px', 
                 fontsize=10, fontweight='bold')
    ax4.set_xticks([])
    ax4.set_yticks([])
    
    # Highlight restored pixels
    if original.size <= 400:
        restored_mask = (eroded == 0) & (opened == 255)
        for i in range(opened.shape[0]):
            for j in range(opened.shape[1]):
                if restored_mask[i, j]:
                    rect = Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                   linewidth=1.5, edgecolor='green', facecolor='none', linestyle='-')
                    ax4.add_patch(rect)
    
    # Highlight permanently removed (was white in original, still black in opening)
    if original.size <= 400:
        perma_removed = (original == 255) & (opened == 0)
        for i in range(opened.shape[0]):
            for j in range(opened.shape[1]):
                if perma_removed[i, j]:
                    rect = Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                   linewidth=2, edgecolor='red', facecolor='red', alpha=0.3)
                    ax4.add_patch(rect)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def test_with_custom_image():
    """Load and apply opening to a custom image"""
    
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.gif"),
            ("All files", "*.*")
        ]
    )
    
    if not file_path:
        print("No file selected.")
        return
    
    try:
        pil_image = Image.open(file_path)
        print(f"\nLoaded: {os.path.basename(file_path)}")
        print(f"Original mode: {pil_image.mode}")
        print(f"Original size: {pil_image.size}")
        
        original_array = np.array(pil_image)
        if len(original_array.shape) == 3:
            print("Converting to grayscale and binarizing...")
            grayscale = convert_to_grayscale(original_array)
            binary = ensure_binary(grayscale)
        else:
            binary = ensure_binary(original_array)
        
        print(f"Binary image shape: {binary.shape}")
        
        # Choose structuring element
        print("\nChoose structuring element:")
        print("1. 3x3 Square")
        print("2. 3x3 Cross")
        print("3. 5x5 Square")
        print("4. 5x5 Cross")
        print("5. Custom size")
        
        se_choice = input("Select (1-5): ").strip()
        
        if se_choice == '1':
            se = create_structuring_element(3, "square")
        elif se_choice == '2':
            se = create_structuring_element(3, "cross")
        elif se_choice == '3':
            se = create_structuring_element(5, "square")
        elif se_choice == '4':
            se = create_structuring_element(5, "cross")
        elif se_choice == '5':
            while True:
                try:
                    size = int(input("Enter SE size (odd number ≥ 3): ").strip())
                    if size >= 3 and size % 2 == 1:
                        break
                    else:
                        print("Size must be an odd number ≥ 3")
                except ValueError:
                    print("Please enter a valid number")
            shape = input("Shape (square/cross): ").strip().lower()
            if shape not in ['square', 'cross']:
                print("Invalid shape, using 'square'")
                shape = 'square'
            se = create_structuring_element(size, shape)
        else:
            print("Invalid choice, using 3x3 square")
            se = create_structuring_element(3, "square")
        
        # Apply opening (and get intermediate erosion result)
        eroded = erode(binary, se)
        opened = opening(binary, se)
        
        print(f"\nOpening Results:")
        print(f"  SE shape: {se.shape}")
        print(f"  White pixels before: {np.sum(binary == 255)}")
        print(f"  After erosion: {np.sum(eroded == 255)} (removed {np.sum(binary == 255) - np.sum(eroded == 255)})")
        print(f"  After opening: {np.sum(opened == 255)} (restored {np.sum(opened == 255) - np.sum(eroded == 255)})")
        print(f"  Permanently removed: {np.sum(binary == 255) - np.sum(opened == 255)}")
        
        # Visualize
        visualize_opening(binary, eroded, opened, se, f"Opening of {os.path.basename(file_path)}")
        
        # Save option
        save_option = input("\nDo you want to save the opened image? (y/n): ").lower().strip()
        if save_option == 'y':
            save_path = filedialog.asksaveasfilename(
                title="Save opened image",
                defaultextension=".png",
                filetypes=[("PNG file", "*.png"), ("JPEG file", "*.jpg"), ("BMP file", "*.bmp")]
            )
            if save_path:
                Image.fromarray(opened).save(save_path)
                print(f"Image saved to: {save_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def demo_with_test_patterns():
    """Demonstrate opening with various test patterns"""
    
    while True:
        print("\n" + "="*60)
        print("MORPHOLOGICAL OPENING VISUALIZER")
        print("Opening = Erosion → Dilation")
        print("Using: processing.morphology.binary_morphology.opening")
        print("="*60)
        print("1. Upload and open your own image")
        print("2. Test salt noise removal")
        print("3. Test thin line removal")
        print("4. Test corner smoothing")
        print("5. Compare square vs cross SE")
        print("6. Compare different SE sizes")
        print("7. Idempotence demonstration (opening twice)")
        print("8. Interactive opening explorer")
        print("9. Erosion vs Opening comparison")
        print("10. Show function source code")
        print("11. Exit")
        
        choice = input("\nSelect option (1-11): ").strip()
        
        if choice == '1':
            test_with_custom_image()
        
        elif choice == '2':
            # Salt noise removal
            print("\n" + "-"*40)
            print("SALT NOISE REMOVAL")
            print("Opening removes isolated white pixels (salt noise)")
            print("-"*40)
            
            # Create image with salt noise
            image = np.zeros((10, 12), dtype=np.uint8)
            image[2:8, 3:9] = 255  # Main 6x6 object
            
            # Add salt noise
            noise_positions = [(0, 5), (9, 3), (5, 10), (0, 0), (9, 8), (3, 0), (7, 11)]
            for pos in noise_positions:
                image[pos] = 255
            
            print("Original (main object + salt noise):")
            print(image)
            print(f"\nNoise positions: {noise_positions}")
            
            se = create_structuring_element(3, "square")
            eroded = erode(image, se)
            opened = opening(image, se)
            
            # Check noise removal
            noise_removed = 0
            for pos in noise_positions:
                if opened[pos] == 0:
                    noise_removed += 1
            
            print(f"\nNoise pixels removed: {noise_removed}/{len(noise_positions)}")
            print(f"Main object preserved: {np.sum(opened[2:8, 3:9] == 255) > 0}")
            
            visualize_opening(image, eroded, opened, se, 
                            f"Salt Noise Removal (removed {noise_removed}/{len(noise_positions)} noise pixels)")
            
            print("\nWhy it works:")
            print("  1. Erosion: Removes isolated white pixels (they don't fit the SE)")
            print("  2. Dilation: Restores the main object (which survived erosion)")
            print("  3. Noise pixels: Gone forever (erosion removed them, nothing to dilate)")
        
        elif choice == '3':
            # Thin line removal
            print("\n" + "-"*40)
            print("THIN LINE REMOVAL")
            print("Opening removes thin structures")
            print("-"*40)
            
            # Create image with thin lines
            image = np.zeros((10, 10), dtype=np.uint8)
            image[3, :] = 255  # Horizontal thin line (1px)
            image[:, 6] = 255  # Vertical thin line (1px)
            image[1:3, 1:3] = 255  # Small 2x2 block
            
            print("Original (thin lines + small block):")
            print(image)
            
            se_configs = [
                ("3x3 Square", create_structuring_element(3, "square")),
                ("5x5 Square", create_structuring_element(5, "square")),
            ]
            
            fig, axes = plt.subplots(2, 3, figsize=(16, 9))
            
            # Original
            axes[0, 0].imshow(image, cmap='gray', vmin=0, vmax=255)
            axes[0, 0].set_title(f'Original\nWhite: {np.sum(image==255)} px', fontsize=10)
            axes[0, 0].axis('off')
            
            for idx, (name, se) in enumerate(se_configs):
                eroded = erode(image, se)
                opened = opening(image, se)
                
                axes[0, idx + 1].imshow(eroded, cmap='gray', vmin=0, vmax=255)
                axes[0, idx + 1].set_title(f'{name}\nErosion\nWhite: {np.sum(eroded==255)} px', fontsize=10)
                axes[0, idx + 1].axis('off')
                
                axes[1, idx + 1].imshow(opened, cmap='gray', vmin=0, vmax=255)
                remaining = np.sum(opened == 255)
                axes[1, idx + 1].set_title(f'{name}\nOpening\nWhite: {remaining} px', fontsize=10)
                axes[1, idx + 1].axis('off')
                
                # Check what survived
                thin_lines_survived = (opened[3, :6] == 255).any() or (opened[:6, 6] == 255).any()
                small_block_survived = np.sum(opened[1:3, 1:3] == 255) > 0
                print(f"\n{name}:")
                print(f"  Thin lines survive: {thin_lines_survived}")
                print(f"  Small 2x2 block survives: {small_block_survived}")
                print(f"  Final white pixels: {remaining}")
            
            plt.suptitle('Thin Line Removal by Opening', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            print("\nKey Insight:")
            print("  Opening removes structures thinner than the SE")
            print("  Only objects that survive erosion are restored")
        
        elif choice == '4':
            # Corner smoothing
            print("\n" + "-"*40)
            print("CORNER SMOOTHING")
            print("Opening rounds sharp corners")
            print("-"*40)
            
            # Create square with sharp corners
            image = np.zeros((9, 9), dtype=np.uint8)
            image[1:8, 1:8] = 255  # 7x7 square
            
            print("Original 7x7 square (sharp corners):")
            print(image)
            
            se_square = create_structuring_element(3, "square")
            se_cross = create_structuring_element(3, "cross")
            
            eroded_sq = erode(image, se_square)
            opened_sq = opening(image, se_square)
            
            eroded_cr = erode(image, se_cross)
            opened_cr = opening(image, se_cross)
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 9))
            
            # Square SE
            axes[0, 0].imshow(image, cmap='gray', vmin=0, vmax=255)
            axes[0, 0].set_title('Original\n7x7 Square', fontsize=10)
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(eroded_sq, cmap='gray', vmin=0, vmax=255)
            axes[0, 1].set_title(f'Erosion (Square SE)\n→ 5x5 Square', fontsize=10)
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(opened_sq, cmap='gray', vmin=0, vmax=255)
            axes[0, 2].set_title(f'Opening (Square SE)\nCorners slightly rounded', fontsize=10)
            axes[0, 2].axis('off')
            
            # Cross SE
            axes[1, 0].imshow(image, cmap='gray', vmin=0, vmax=255)
            axes[1, 0].set_title('Original\n7x7 Square', fontsize=10)
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(eroded_cr, cmap='gray', vmin=0, vmax=255)
            axes[1, 1].set_title(f'Erosion (Cross SE)\nDifferent pattern', fontsize=10)
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow(opened_cr, cmap='gray', vmin=0, vmax=255)
            axes[1, 2].set_title(f'Opening (Cross SE)\nMore rounded corners', fontsize=10)
            axes[1, 2].axis('off')
            
            plt.suptitle('Corner Smoothing by Opening', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            print("\nCorner Analysis:")
            corners = [(1,1), (1,7), (7,1), (7,7)]
            for se_name, opened in [("Square SE", opened_sq), ("Cross SE", opened_cr)]:
                removed_corners = sum(1 for c in corners if opened[c] == 0)
                print(f"  {se_name}: {removed_corners}/4 corners removed")
        
        elif choice == '5':
            # Compare square vs cross SE
            print("\n" + "-"*40)
            print("SQUARE VS CROSS SE")
            print("-"*40)
            
            # Test patterns
            patterns = {
                'With Salt Noise': np.zeros((9, 9), dtype=np.uint8),
                'With Thin Lines': np.zeros((7, 7), dtype=np.uint8),
                'With Holes': np.zeros((7, 7), dtype=np.uint8),
            }
            
            # Salt noise pattern
            patterns['With Salt Noise'][2:7, 2:7] = 255
            patterns['With Salt Noise'][[0, 0, 8], [1, 5, 7]] = 255
            
            # Thin lines pattern
            patterns['With Thin Lines'][1:6, 1:6] = 255
            patterns['With Thin Lines'][3, :] = 255
            
            # Holes pattern
            patterns['With Holes'][1:6, 1:6] = 255
            patterns['With Holes'][2:5, 2:5] = 0
            
            se_square = create_structuring_element(3, "square")
            se_cross = create_structuring_element(3, "cross")
            
            fig, axes = plt.subplots(len(patterns), 3, figsize=(12, 12))
            
            for idx, (name, pattern) in enumerate(patterns.items()):
                opened_sq = opening(pattern, se_square)
                opened_cr = opening(pattern, se_cross)
                
                axes[idx, 0].imshow(pattern, cmap='gray', vmin=0, vmax=255)
                axes[idx, 0].set_title(f'{name}\nOriginal\nWhite: {np.sum(pattern==255)}', fontsize=9)
                axes[idx, 0].axis('off')
                
                axes[idx, 1].imshow(opened_sq, cmap='gray', vmin=0, vmax=255)
                white_sq = np.sum(opened_sq == 255)
                axes[idx, 1].set_title(f'Square SE\nWhite: {white_sq}\nKept: {white_sq/np.sum(pattern==255)*100:.0f}%', fontsize=9)
                axes[idx, 1].axis('off')
                
                axes[idx, 2].imshow(opened_cr, cmap='gray', vmin=0, vmax=255)
                white_cr = np.sum(opened_cr == 255)
                axes[idx, 2].set_title(f'Cross SE\nWhite: {white_cr}\nKept: {white_cr/np.sum(pattern==255)*100:.0f}%', fontsize=9)
                axes[idx, 2].axis('off')
            
            plt.suptitle('Square vs Cross SE: Opening Comparison', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            print("\nSquare SE (9 active) vs Cross SE (5 active):")
            print("  Square SE: More aggressive removal of small features")
            print("  Cross SE: Preserves more diagonal structures")
        
        elif choice == '6':
            # Compare different SE sizes
            print("\n" + "-"*40)
            print("COMPARING SE SIZES")
            print("-"*40)
            
            # Create pattern with features of different sizes
            image = np.zeros((12, 12), dtype=np.uint8)
            image[3:9, 3:9] = 255  # 6x6 main object
            image[1, 1] = 255  # Single pixel noise
            image[10, 10] = 255  # Single pixel noise
            image[5:7, 10] = 255  # 2x1 thin feature
            image[0:3, 8] = 255  # 3x1 feature
            
            print("Original (various feature sizes):")
            print(image)
            
            se_sizes = [3, 5, 7]
            
            fig, axes = plt.subplots(1, len(se_sizes) + 1, figsize=(18, 4))
            
            axes[0].imshow(image, cmap='gray', vmin=0, vmax=255)
            axes[0].set_title(f'Original\nWhite: {np.sum(image==255)} px', fontsize=10)
            axes[0].axis('off')
            
            for idx, se_size in enumerate(se_sizes):
                se = create_structuring_element(se_size, "square")
                opened = opening(image, se)
                
                white_remaining = np.sum(opened == 255)
                removed = np.sum(image == 255) - white_remaining
                
                axes[idx + 1].imshow(opened, cmap='gray', vmin=0, vmax=255)
                axes[idx + 1].set_title(f'{se_size}x{se_size} SE\nRemaining: {white_remaining} px\nRemoved: {removed} px', fontsize=10)
                axes[idx + 1].axis('off')
            
            plt.suptitle('Effect of SE Size on Opening', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            print("\nSize Comparison:")
            for se_size in se_sizes:
                se = create_structuring_element(se_size, "square")
                opened = opening(image, se)
                remaining = np.sum(opened == 255)
                print(f"  {se_size}x{se_size} SE: {remaining} white pixels remain")
                print(f"    Features smaller than {se_size}x{se_size} are removed")
        
        elif choice == '7':
            # Idempotence demonstration
            print("\n" + "-"*40)
            print("IDEMPOTENCE: Opening Twice = Opening Once")
            print("-"*40)
            
            # Create test image
            image = np.zeros((8, 8), dtype=np.uint8)
            image[1:7, 1:7] = 255  # 6x6 square
            image[0, 3] = 255  # Noise
            image[7, 5] = 255  # Noise
            
            print("Original:")
            print(image)
            
            se = create_structuring_element(3, "square")
            
            # Apply opening once
            opened1 = opening(image, se)
            
            # Apply opening twice
            opened2 = opening(opened1, se)
            
            # Check if identical
            identical = np.array_equal(opened1, opened2)
            
            print(f"\nOpening once and twice are identical: {identical}")
            print("\nAfter first opening:")
            print(opened1)
            print("\nAfter second opening:")
            print(opened2)
            
            # Visual
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            axes[0].imshow(image, cmap='gray', vmin=0, vmax=255)
            axes[0].set_title(f'Original\nWhite: {np.sum(image==255)} px', fontsize=11)
            axes[0].axis('off')
            
            axes[1].imshow(opened1, cmap='gray', vmin=0, vmax=255)
            axes[1].set_title(f'Opening Once\nWhite: {np.sum(opened1==255)} px', fontsize=11)
            axes[1].axis('off')
            
            axes[2].imshow(opened2, cmap='gray', vmin=0, vmax=255)
            axes[2].set_title(f'Opening Twice\nWhite: {np.sum(opened2==255)} px\nIdentical: {identical}', fontsize=11)
            axes[2].axis('off')
            
            plt.suptitle('Idempotence: Opening Twice = Opening Once', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            print("\nWhy is opening idempotent?")
            print("  1st Opening: Removes all features smaller than SE")
            print("  2nd Opening: No small features left to remove → No change")
        
        elif choice == '8':
            # Interactive opening explorer
            print("\n" + "-"*40)
            print("INTERACTIVE OPENING EXPLORER")
            print("-"*40)
            
            # Create test pattern
            pattern = np.zeros((10, 10), dtype=np.uint8)
            pattern[2:8, 2:8] = 255  # Main object
            pattern[0, 0] = 255  # Corner noise
            pattern[0, 9] = 255  # Corner noise
            pattern[9, 5] = 255  # Edge noise
            pattern[5, 9] = 255  # Edge noise
            
            print("Original pattern (main object + noise):")
            print(pattern)
            
            while True:
                print("\nEnter SE parameters (-1 to quit):")
                try:
                    se_size = int(input("  SE size (odd, ≥ 3): ").strip())
                    if se_size == -1:
                        break
                    se_shape = input("  SE shape (square/cross): ").strip().lower()
                    
                    if se_size >= 3 and se_size % 2 == 1 and se_shape in ['square', 'cross']:
                        se = create_structuring_element(se_size, se_shape)
                        eroded = erode(pattern, se)
                        opened = opening(pattern, se)
                        
                        noise_removed = 0
                        noise_positions = [(0,0), (0,9), (9,5), (5,9)]
                        for pos in noise_positions:
                            if opened[pos] == 0:
                                noise_removed += 1
                        
                        print(f"\n{se_size}x{se_size} {se_shape} SE:")
                        print(f"  Noise removed: {noise_removed}/{len(noise_positions)}")
                        print(f"  Main object preserved: {np.sum(opened[2:8, 2:8] == 255) > 0}")
                        
                        visualize_opening(pattern, eroded, opened, se, 
                                        f"Interactive: {se_size}x{se_size} {se_shape} - Noise removed: {noise_removed}/4")
                    else:
                        print("Invalid parameters")
                except ValueError:
                    print("Please enter valid numbers")
        
        elif choice == '9':
            # Erosion vs Opening comparison
            print("\n" + "-"*40)
            print("EROSION VS OPENING")
            print("Opening = Erosion + Dilation")
            print("-"*40)
            
            # Create image with noise and main object
            image = np.zeros((8, 8), dtype=np.uint8)
            image[1:7, 1:7] = 255  # 6x6 main object
            image[0, 2] = 255  # Noise
            image[0, 5] = 255  # Noise
            image[7, 3] = 255  # Noise
            
            print("Original (main object + noise):")
            print(image)
            
            se = create_structuring_element(3, "square")
            eroded = erode(image, se)
            opened = opening(image, se)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            axes[0].imshow(image, cmap='gray', vmin=0, vmax=255)
            axes[0].set_title(f'Original\nWhite: {np.sum(image==255)} px\nMain object + noise', fontsize=10)
            axes[0].axis('off')
            
            axes[1].imshow(eroded, cmap='gray', vmin=0, vmax=255)
            axes[1].set_title(f'Erosion Only\nWhite: {np.sum(eroded==255)} px\nObject shrinks, noise removed', fontsize=10)
            axes[1].axis('off')
            
            axes[2].imshow(opened, cmap='gray', vmin=0, vmax=255)
            axes[2].set_title(f'Opening (Erosion+Dilation)\nWhite: {np.sum(opened==255)} px\nObject restored, noise gone', fontsize=10)
            axes[2].axis('off')
            
            plt.suptitle('Erosion vs Opening: Opening Preserves Object Size', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            print("\nKey Difference:")
            print(f"  Erosion: {np.sum(image==255)} → {np.sum(eroded==255)} px (object SHRINKS)")
            print(f"  Opening: {np.sum(image==255)} → {np.sum(opened==255)} px (object size PRESERVED)")
            print("\n  Erosion removes noise but also shrinks the main object")
            print("  Opening removes noise while PRESERVING the main object size")
        
        elif choice == '10':
            import inspect
            print("\n" + "="*60)
            print("Function source code:")
            print("="*60)
            print(inspect.getsource(opening))
            print("="*60)
            print(f"Function location: {inspect.getfile(opening)}")
        
        elif choice == '11':
            print("Goodbye!")
            break
        
        else:
            print("Invalid option. Please try again.")


def quick_demo():
    """Quick demonstration of opening"""
    
    print("\n" + "="*60)
    print("QUICK DEMO - opening()")
    print("="*60)
    
    # Create image with salt noise
    image = np.zeros((7, 7), dtype=np.uint8)
    image[1:6, 1:6] = 255  # 5x5 main object
    image[0, 0] = 255  # Noise
    image[0, 6] = 255  # Noise
    image[6, 3] = 255  # Noise
    
    se = create_structuring_element(3, "square")
    eroded = erode(image, se)
    opened = opening(image, se)
    
    print("\nOriginal (with salt noise):")
    print(image)
    print(f"White pixels: {np.sum(image==255)}")
    
    print("\nAfter erosion:")
    print(eroded)
    print(f"White pixels: {np.sum(eroded==255)} (noise removed, object shrinks)")
    
    print("\nAfter dilation (Opening result):")
    print(opened)
    print(f"White pixels: {np.sum(opened==255)} (object restored, noise gone)")
    
    print("\nKey Insight: Opening = Remove noise while preserving object size!")
    
    # Visual
    visualize_opening(image, eroded, opened, se, "Opening Quick Demo: Remove Noise, Preserve Object")


if __name__ == '__main__':
    try:
        import inspect
        print(f"Successfully imported from:")
        print(f"  opening: {inspect.getfile(opening)}")
        print(f"  erode: {inspect.getfile(erode)}")
        print(f"  dilate: {inspect.getfile(dilate)}")
    except ImportError as e:
        print(f"Error importing function: {e}")
        sys.exit(1)
    
    quick_demo()
    demo_with_test_patterns()