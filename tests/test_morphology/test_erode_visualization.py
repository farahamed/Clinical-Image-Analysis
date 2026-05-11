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
from processing.morphology.binary_morphology import erode, create_structuring_element, convert_to_grayscale, ensure_binary


def visualize_erosion(original, eroded, structuring_element, title="Erosion Result"):
    """Visualize original vs eroded images with SE display"""
    
    fig = plt.figure(figsize=(16, 5))
    
    # Original image
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(original, cmap='gray', vmin=0, vmax=255)
    ax1.set_title(f'Original Image\nShape: {original.shape}', fontsize=11, fontweight='bold')
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Add grid and pixel values for small images
    if original.size <= 100:
        for i in range(original.shape[0] + 1):
            ax1.axhline(y=i - 0.5, color='blue', linewidth=0.5, alpha=0.3)
            ax1.axvline(x=i - 0.5, color='blue', linewidth=0.5, alpha=0.3)
        for i in range(original.shape[0]):
            for j in range(original.shape[1]):
                color = 'white' if original[i, j] < 128 else 'black'
                ax1.text(j, i, str(original[i, j]), ha='center', va='center', 
                        color=color, fontsize=7, fontweight='bold')
    
    # Structuring Element
    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(structuring_element, cmap='gray', vmin=0, vmax=1)
    ax2.set_title(f'Structuring Element\nShape: {structuring_element.shape}', fontsize=11, fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Add grid for SE
    for i in range(structuring_element.shape[0] + 1):
        ax2.axhline(y=i - 0.5, color='green', linewidth=1, alpha=0.5)
        ax2.axvline(x=i - 0.5, color='green', linewidth=1, alpha=0.5)
    for i in range(structuring_element.shape[0]):
        for j in range(structuring_element.shape[1]):
            color = 'white' if structuring_element[i, j] < 0.5 else 'black'
            ax2.text(j, i, str(structuring_element[i, j]), ha='center', va='center', 
                    color=color, fontsize=9, fontweight='bold')
    
    # Eroded image
    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(eroded, cmap='gray', vmin=0, vmax=255)
    
    # Count statistics
    white_before = np.sum(original == 255)
    white_after = np.sum(eroded == 255)
    white_removed = white_before - white_after
    
    ax3.set_title(f'Eroded Image\nShape: {eroded.shape}\nWhite px: {white_before} → {white_after} (-{white_removed})', 
                 fontsize=11, fontweight='bold')
    ax3.set_xticks([])
    ax3.set_yticks([])
    
    # Add grid for small images
    if eroded.size <= 100:
        for i in range(eroded.shape[0] + 1):
            ax3.axhline(y=i - 0.5, color='blue', linewidth=0.5, alpha=0.3)
            ax3.axvline(x=i - 0.5, color='blue', linewidth=0.5, alpha=0.3)
        for i in range(eroded.shape[0]):
            for j in range(eroded.shape[1]):
                if eroded[i, j] == 255:
                    ax3.text(j, i, '255', ha='center', va='center', color='black', fontsize=7, fontweight='bold')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def test_with_custom_image():
    """Load and erode a custom image"""
    
    # Create a root window and hide it
    root = tk.Tk()
    root.withdraw()
    
    # Open file dialog
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
        # Load the image
        pil_image = Image.open(file_path)
        print(f"\nLoaded: {os.path.basename(file_path)}")
        print(f"Original mode: {pil_image.mode}")
        print(f"Original size: {pil_image.size}")
        
        # Convert to numpy array and ensure binary
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
        
        # Apply erosion
        eroded = erode(binary, se)
        
        print(f"\nErosion Results:")
        print(f"  SE shape: {se.shape}")
        print(f"  White pixels before: {np.sum(binary == 255)}")
        print(f"  White pixels after: {np.sum(eroded == 255)}")
        print(f"  Pixels removed: {np.sum(binary == 255) - np.sum(eroded == 255)}")
        
        # Visualize
        visualize_erosion(binary, eroded, se, f"Erosion of {os.path.basename(file_path)}")
        
        # Save option
        save_option = input("\nDo you want to save the eroded image? (y/n): ").lower().strip()
        if save_option == 'y':
            save_path = filedialog.asksaveasfilename(
                title="Save eroded image",
                defaultextension=".png",
                filetypes=[("PNG file", "*.png"), ("JPEG file", "*.jpg"), ("BMP file", "*.bmp")]
            )
            if save_path:
                Image.fromarray(eroded).save(save_path)
                print(f"Image saved to: {save_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def demo_with_test_patterns():
    """Demonstrate erosion with various test patterns"""
    
    while True:
        print("\n" + "="*60)
        print("EROSION VISUALIZER")
        print("Using: processing.morphology.binary_morphology.erode")
        print("="*60)
        print("1. Upload and erode your own image")
        print("2. Test with square pattern")
        print("3. Test with cross pattern")
        print("4. Test removing thin lines")
        print("5. Test separating objects")
        print("6. Compare different SE sizes")
        print("7. Compare square vs cross SE")
        print("8. Interactive erosion explorer")
        print("9. Show function source code")
        print("10. Exit")
        
        choice = input("\nSelect option (1-10): ").strip()
        
        if choice == '1':
            test_with_custom_image()
        
        elif choice == '2':
            # Square pattern test
            print("\n" + "-"*40)
            print("SQUARE PATTERN EROSION")
            print("-"*40)
            
            # Create a white square in black background
            size = int(input("Image size (default=7): ").strip() or "7")
            square_size = int(input("Square size (default=3): ").strip() or "3")
            
            # Create image
            image = np.zeros((size, size), dtype=np.uint8)
            start = (size - square_size) // 2
            image[start:start+square_size, start:start+square_size] = 255
            
            print(f"\nOriginal image ({size}x{size} with {square_size}x{square_size} white square):")
            print(image)
            
            # Get SE
            while True:
                try:
                    se_size = int(input("\nSE size (odd, default=3): ").strip() or "3")
                    if se_size >= 3 and se_size % 2 == 1:
                        break
                    else:
                        print("Size must be an odd number ≥ 3")
                except ValueError:
                    print("Please enter a valid number")
            
            se_shape = input("SE shape (square/cross, default=square): ").strip().lower() or "square"
            se = create_structuring_element(se_size, se_shape)
            
            # Apply erosion
            eroded = erode(image, se)
            
            print(f"\nSE ({se_size}x{se_size} {se_shape}):")
            print(se)
            print(f"\nEroded image:")
            print(eroded)
            
            # Statistics
            white_before = np.sum(image == 255)
            white_after = np.sum(eroded == 255)
            print(f"\nWhite pixels: {white_before} → {white_after}")
            if white_after > 0:
                print(f"Square shrunk from {square_size}x{square_size} to approximately {int(np.sqrt(white_after))}x{int(np.sqrt(white_after))}")
            
            visualize_erosion(image, eroded, se, f"Erosion of {square_size}x{square_size} Square")
        
        elif choice == '3':
            # Cross pattern test
            print("\n" + "-"*40)
            print("CROSS PATTERN EROSION")
            print("-"*40)
            
            # Create a cross pattern
            size = 7
            cross = np.zeros((size, size), dtype=np.uint8)
            cross[3, :] = 255  # Horizontal line
            cross[:, 3] = 255  # Vertical line
            
            print("Original cross pattern:")
            print(cross)
            
            # Get SEs
            se_square = create_structuring_element(3, "square")
            se_cross = create_structuring_element(3, "cross")
            
            eroded_square = erode(cross, se_square)
            eroded_cross_se = erode(cross, se_cross)
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            
            # Original
            axes[0, 0].imshow(cross, cmap='gray', vmin=0, vmax=255)
            axes[0, 0].set_title(f'Original Cross\n{cross.shape}', fontsize=11)
            axes[0, 0].axis('off')
            
            # Square SE
            axes[0, 1].imshow(se_square, cmap='gray', vmin=0, vmax=1)
            axes[0, 1].set_title('3x3 Square SE', fontsize=11)
            axes[0, 1].axis('off')
            for i in range(4):
                axes[0, 1].axhline(y=i-0.5, color='green', linewidth=1, alpha=0.5)
                axes[0, 1].axvline(x=i-0.5, color='green', linewidth=1, alpha=0.5)
            
            # Eroded with square
            axes[0, 2].imshow(eroded_square, cmap='gray', vmin=0, vmax=255)
            axes[0, 2].set_title(f'Eroded (3x3 Square SE)\nWhite: {np.sum(cross==255)} → {np.sum(eroded_square==255)}', fontsize=11)
            axes[0, 2].axis('off')
            
            # Original (again)
            axes[1, 0].imshow(cross, cmap='gray', vmin=0, vmax=255)
            axes[1, 0].set_title('Original Cross', fontsize=11)
            axes[1, 0].axis('off')
            
            # Cross SE
            axes[1, 1].imshow(se_cross, cmap='gray', vmin=0, vmax=1)
            axes[1, 1].set_title('3x3 Cross SE', fontsize=11)
            axes[1, 1].axis('off')
            for i in range(4):
                axes[1, 1].axhline(y=i-0.5, color='green', linewidth=1, alpha=0.5)
                axes[1, 1].axvline(x=i-0.5, color='green', linewidth=1, alpha=0.5)
            
            # Eroded with cross
            axes[1, 2].imshow(eroded_cross_se, cmap='gray', vmin=0, vmax=255)
            axes[1, 2].set_title(f'Eroded (3x3 Cross SE)\nWhite: {np.sum(cross==255)} → {np.sum(eroded_cross_se==255)}', fontsize=11)
            axes[1, 2].axis('off')
            
            plt.suptitle('Erosion of Cross Pattern with Different SEs', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            print("\nResults:")
            print(f"  Original white pixels: {np.sum(cross==255)}")
            print(f"  After 3x3 Square SE: {np.sum(eroded_square==255)}")
            print(f"  After 3x3 Cross SE: {np.sum(eroded_cross_se==255)}")
            print(f"\n  Note: Cross SE has fewer active elements (5 vs 9)")
            print(f"  so it erodes less aggressively!")
        
        elif choice == '4':
            # Remove thin lines
            print("\n" + "-"*40)
            print("REMOVING THIN LINES")
            print("-"*40)
            
            # Create pattern with thin lines
            pattern = np.zeros((9, 9), dtype=np.uint8)
            pattern[1, :] = 255  # 1px horizontal line (top)
            pattern[4, :] = 255  # 1px horizontal line (middle)
            pattern[7, :] = 255  # 1px horizontal line (bottom)
            pattern[:, 4] = 255  # 1px vertical line (will create intersections)
            
            print("Original pattern (thin lines):")
            print(pattern)
            
            se_sizes = [3, 5]
            fig, axes = plt.subplots(1, len(se_sizes) + 1, figsize=(15, 4))
            
            # Original
            axes[0].imshow(pattern, cmap='gray', vmin=0, vmax=255)
            axes[0].set_title(f'Original\nThin Lines\n{pattern.shape}', fontsize=10)
            axes[0].axis('off')
            
            for idx, se_size in enumerate(se_sizes):
                se = create_structuring_element(se_size, "square")
                eroded = erode(pattern, se)
                
                axes[idx + 1].imshow(eroded, cmap='gray', vmin=0, vmax=255)
                axes[idx + 1].set_title(f'{se_size}x{se_size} Square SE\nWhite: {np.sum(eroded==255)} px remain', fontsize=10)
                axes[idx + 1].axis('off')
            
            plt.suptitle('Thin Line Removal by Erosion', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            # More details
            print("\nDetailed Results:")
            for se_size in se_sizes:
                se = create_structuring_element(se_size, "square")
                eroded = erode(pattern, se)
                print(f"\n{se_size}x{se_size} Square SE:")
                print(f"  White pixels before: {np.sum(pattern==255)}")
                print(f"  White pixels after: {np.sum(eroded==255)}")
                print(f"  Pixels removed: {np.sum(pattern==255) - np.sum(eroded==255)}")
        
        elif choice == '5':
            # Separate objects
            print("\n" + "-"*40)
            print("SEPARATING OBJECTS")
            print("-"*40)
            
            # Create two close squares
            image = np.zeros((10, 14), dtype=np.uint8)
            image[2:7, 2:6] = 255  # Left 5x4 square
            image[2:7, 8:12] = 255  # Right 5x4 square
            # Gap at columns 6-7 (1px gap)
            
            print("Original (two squares with 1px gap):")
            print(image)
            
            se = create_structuring_element(3, "square")
            eroded = erode(image, se)
            
            print("\nAfter erosion with 3x3 square SE:")
            print(eroded)
            
            # Check if they're still connected
            connected = False
            for i in range(eroded.shape[0]):
                for j in range(eroded.shape[1] - 1):
                    if eroded[i, j] == 255 and eroded[i, j+1] == 255:
                        if j >= 5 and j <= 7:  # Around gap area
                            connected = True
            
            print(f"\nObjects still connected? {connected}")
            print(f"Gap width increased from 1 to 3 pixels")
            
            # Visual
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            axes[0].imshow(image, cmap='gray', vmin=0, vmax=255)
            axes[0].set_title('Original\nTwo Squares (1px gap)', fontsize=11)
            axes[0].axis('off')
            
            axes[1].imshow(se, cmap='gray', vmin=0, vmax=1)
            axes[1].set_title('3x3 Square SE', fontsize=11)
            axes[1].axis('off')
            for i in range(4):
                axes[1].axhline(y=i-0.5, color='green', linewidth=1, alpha=0.5)
                axes[1].axvline(x=i-0.5, color='green', linewidth=1, alpha=0.5)
            
            axes[2].imshow(eroded, cmap='gray', vmin=0, vmax=255)
            axes[2].set_title(f'Eroded\nGap increased to 3px\nSquares separated', fontsize=11)
            axes[2].axis('off')
            
            plt.suptitle('Erosion Increases Gap Between Objects', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
        
        elif choice == '6':
            # Compare different SE sizes
            print("\n" + "-"*40)
            print("COMPARING DIFFERENT SE SIZES")
            print("-"*40)
            
            # Create test pattern
            test = np.zeros((9, 9), dtype=np.uint8)
            test[2:7, 2:7] = 255  # 5x5 white square
            
            print("Original 5x5 white square in 9x9 image")
            print(test)
            
            se_sizes = [3, 5, 7]
            fig, axes = plt.subplots(1, len(se_sizes) + 1, figsize=(16, 4))
            
            # Original
            axes[0].imshow(test, cmap='gray', vmin=0, vmax=255)
            axes[0].set_title(f'Original\n5x5 Square\nWhite: {np.sum(test==255)} px', fontsize=10)
            axes[0].axis('off')
            
            for idx, se_size in enumerate(se_sizes):
                se = create_structuring_element(se_size, "square")
                eroded = erode(test, se)
                
                axes[idx + 1].imshow(eroded, cmap='gray', vmin=0, vmax=255)
                
                remaining = np.sum(eroded == 255)
                if remaining > 0:
                    size_approx = int(np.sqrt(remaining))
                    axes[idx + 1].set_title(f'{se_size}x{se_size} Square SE\nRemaining: {remaining} px\n(~{size_approx}x{size_approx} square)', fontsize=10)
                else:
                    axes[idx + 1].set_title(f'{se_size}x{se_size} Square SE\nRemaining: 0 px\n(Object vanished!)', fontsize=10)
                axes[idx + 1].axis('off')
            
            plt.suptitle('Effect of SE Size on Erosion', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            print("\nShape Transformation:")
            print(f"  Original: 5x5 = 25 white pixels")
            for se_size in se_sizes:
                se = create_structuring_element(se_size, "square")
                eroded = erode(test, se)
                remaining = np.sum(eroded == 255)
                if remaining > 0:
                    new_size = int(np.sqrt(remaining))
                    print(f"  {se_size}x{se_size} SE: {remaining} white pixels ({new_size}x{new_size} square)")
                else:
                    print(f"  {se_size}x{se_size} SE: 0 white pixels (vanished)")
        
        elif choice == '7':
            # Compare square vs cross SE
            print("\n" + "-"*40)
            print("SQUARE VS CROSS SE COMPARISON")
            print("-"*40)
            
            # Create various patterns
            patterns = {
                'Solid Square': np.ones((7, 7), dtype=np.uint8) * 255,
                'Hollow Square': np.zeros((7, 7), dtype=np.uint8),
                'Single Pixel': np.zeros((7, 7), dtype=np.uint8),
            }
            patterns['Hollow Square'][1:6, 1:6] = 255
            patterns['Hollow Square'][2:5, 2:5] = 0
            patterns['Single Pixel'][3, 3] = 255
            
            se_square = create_structuring_element(3, "square")
            se_cross = create_structuring_element(3, "cross")
            
            fig, axes = plt.subplots(len(patterns), 3, figsize=(12, 12))
            
            for idx, (name, pattern) in enumerate(patterns.items()):
                eroded_sq = erode(pattern, se_square)
                eroded_cr = erode(pattern, se_cross)
                
                # Original
                axes[idx, 0].imshow(pattern, cmap='gray', vmin=0, vmax=255)
                axes[idx, 0].set_title(f'{name}\nOriginal\nWhite: {np.sum(pattern==255)}', fontsize=9)
                axes[idx, 0].axis('off')
                
                # Square SE result
                axes[idx, 1].imshow(eroded_sq, cmap='gray', vmin=0, vmax=255)
                axes[idx, 1].set_title(f'3x3 Square SE\nWhite: {np.sum(eroded_sq==255)}', fontsize=9)
                axes[idx, 1].axis('off')
                
                # Cross SE result
                axes[idx, 2].imshow(eroded_cr, cmap='gray', vmin=0, vmax=255)
                axes[idx, 2].set_title(f'3x3 Cross SE\nWhite: {np.sum(eroded_cr==255)}', fontsize=9)
                axes[idx, 2].axis('off')
            
            plt.suptitle('Square vs Cross SE: Which Erodes More?', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            print("\nComparison Summary:")
            print(f"  Square SE active elements: {np.sum(se_square)}")
            print(f"  Cross SE active elements: {np.sum(se_cross)}")
            print(f"  Square SE is more aggressive (removes more pixels)")
            print(f"  Cross SE preserves more structure")
        
        elif choice == '8':
            # Interactive erosion explorer
            print("\n" + "-"*40)
            print("INTERACTIVE EROSION EXPLORER")
            print("-"*40)
            
            # Create a complex test pattern
            pattern = np.zeros((9, 9), dtype=np.uint8)
            pattern[1:4, 1:4] = 255  # 3x3 square
            pattern[5:8, 5:8] = 255  # 3x3 square
            pattern[1:4, 5:8] = 255  # 3x4 rectangle
            pattern[4, :] = 255  # Horizontal line
            
            print("Original pattern:")
            print(pattern)
            
            while True:
                print("\nCurrent pattern displayed above")
                print("Enter SE parameters (-1 to quit):")
                try:
                    se_size = int(input("  SE size (odd, ≥ 3): ").strip())
                    if se_size == -1:
                        break
                    se_shape = input("  SE shape (square/cross): ").strip().lower()
                    
                    if se_size >= 3 and se_size % 2 == 1 and se_shape in ['square', 'cross']:
                        se = create_structuring_element(se_size, se_shape)
                        eroded = erode(pattern, se)
                        
                        print(f"\n{se_size}x{se_size} {se_shape} SE:")
                        print(f"  White pixels before: {np.sum(pattern==255)}")
                        print(f"  White pixels after: {np.sum(eroded==255)}")
                        print(f"  Removed: {np.sum(pattern==255) - np.sum(eroded==255)}")
                        
                        visualize_erosion(pattern, eroded, se, 
                                        f"Interactive: {se_size}x{se_size} {se_shape} SE")
                    else:
                        print("Invalid parameters. Size must be odd ≥ 3, shape: square/cross")
                except ValueError:
                    print("Please enter valid numbers")
        
        elif choice == '9':
            # Show source code
            import inspect
            print("\n" + "="*60)
            print("Function source code:")
            print("="*60)
            print(inspect.getsource(erode))
            print("="*60)
            print(f"Function location: {inspect.getfile(erode)}")
        
        elif choice == '10':
            print("Goodbye!")
            break
        
        else:
            print("Invalid option. Please try again.")


def quick_demo():
    """Quick demonstration of erosion"""
    
    print("\n" + "="*60)
    print("QUICK DEMO - erode()")
    print("="*60)
    
    # Create test patterns
    square_3x3 = np.array([
        [0, 0, 0, 0, 0],
        [0, 255, 255, 255, 0],
        [0, 255, 255, 255, 0],
        [0, 255, 255, 255, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.uint8)
    
    se_3x3 = create_structuring_element(3, "square")
    se_3x3_cross = create_structuring_element(3, "cross")
    
    # Apply erosion
    eroded_square = erode(square_3x3, se_3x3)
    eroded_cross = erode(square_3x3, se_3x3_cross)
    
    print("\nOriginal 3x3 white square:")
    print(square_3x3)
    print(f"White pixels: {np.sum(square_3x3==255)}")
    
    print("\n3x3 Square SE:")
    print(se_3x3)
    print(f"\nAfter erosion (Square SE):")
    print(eroded_square)
    print(f"White pixels: {np.sum(eroded_square==255)}")
    
    print("\n3x3 Cross SE:")
    print(se_3x3_cross)
    print(f"\nAfter erosion (Cross SE):")
    print(eroded_cross)
    print(f"White pixels: {np.sum(eroded_cross==255)}")
    
    # Visual
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    
    # Row 1: Square SE
    axes[0, 0].imshow(square_3x3, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('Original\n3x3 Square', fontsize=10)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(se_3x3, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('3x3 Square SE', fontsize=10)
    axes[0, 1].axis('off')
    for i in range(4):
        axes[0, 1].axhline(y=i-0.5, color='green', linewidth=1, alpha=0.5)
        axes[0, 1].axvline(x=i-0.5, color='green', linewidth=1, alpha=0.5)
    
    axes[0, 2].imshow(eroded_square, cmap='gray', vmin=0, vmax=255)
    axes[0, 2].set_title(f'Eroded (Square SE)\n1 pixel remains', fontsize=10)
    axes[0, 2].axis('off')
    
    # Row 2: Cross SE
    axes[1, 0].imshow(square_3x3, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title('Original\n3x3 Square', fontsize=10)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(se_3x3_cross, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('3x3 Cross SE', fontsize=10)
    axes[1, 1].axis('off')
    for i in range(4):
        axes[1, 1].axhline(y=i-0.5, color='green', linewidth=1, alpha=0.5)
        axes[1, 1].axvline(x=i-0.5, color='green', linewidth=1, alpha=0.5)
    
    axes[1, 2].imshow(eroded_cross, cmap='gray', vmin=0, vmax=255)
    axes[1, 2].set_title(f'Eroded (Cross SE)\n1 pixel remains', fontsize=10)
    axes[1, 2].axis('off')
    
    plt.suptitle('Erosion Demo: 3x3 White Square', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Verify the import works
    try:
        import inspect
        print(f"Successfully imported from:")
        print(f"  erode: {inspect.getfile(erode)}")
        print(f"  create_structuring_element: {inspect.getfile(create_structuring_element)}")
        print(f"  convert_to_grayscale: {inspect.getfile(convert_to_grayscale)}")
        print(f"  ensure_binary: {inspect.getfile(ensure_binary)}")
    except ImportError as e:
        print(f"Error importing function: {e}")
        print("Make sure the processing module is in your Python path")
        sys.exit(1)
    
    # Show quick demo
    quick_demo()
    
    # Launch interactive mode
    demo_with_test_patterns()