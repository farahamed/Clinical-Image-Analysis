import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from processing.morphology.binary_morphology import convert_to_grayscale


def load_and_convert_image():
    """Open file dialog, load image, apply conversion, and display results"""
    
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
        # Load the image using PIL
        pil_image = Image.open(file_path)
        print(f"\nLoaded: {os.path.basename(file_path)}")
        print(f"Original mode: {pil_image.mode}")
        print(f"Original size: {pil_image.size}")
        
        # Convert PIL image to numpy array
        original_array = np.array(pil_image)
        print(f"Numpy array shape: {original_array.shape}")
        print(f"Numpy array dtype: {original_array.dtype}")
        
        # Apply grayscale conversion
        grayscale_array = convert_to_grayscale(original_array)
        print(f"Grayscale shape: {grayscale_array.shape}")
        print(f"Grayscale dtype: {grayscale_array.dtype}")
        
        # Display original and grayscale images side by side
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Show original image
        if len(original_array.shape) == 3 and original_array.shape[2] == 3:
            axes[0].imshow(original_array)
        else:
            axes[0].imshow(original_array, cmap='gray')
        axes[0].set_title(f'Original\nShape: {original_array.shape}', fontsize=10)
        axes[0].axis('off')
        
        # Show grayscale image
        axes[1].imshow(grayscale_array, cmap='gray')
        axes[1].set_title(f'Grayscale Result\nShape: {grayscale_array.shape}', fontsize=10)
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Ask if user wants to save the result
        save_option = input("\nDo you want to save the grayscale image? (y/n): ").lower().strip()
        if save_option == 'y':
            save_path = filedialog.asksaveasfilename(
                title="Save grayscale image",
                defaultextension=".png",
                filetypes=[
                    ("PNG file", "*.png"),
                    ("JPEG file", "*.jpg"),
                    ("BMP file", "*.bmp"),
                    ("TIFF file", "*.tiff")
                ]
            )
            if save_path:
                Image.fromarray(grayscale_array).save(save_path)
                print(f"Image saved to: {save_path}")
        
    except Exception as e:
        print(f"Error: {e}")


def demo_with_sample_images():
    """Demonstrate the function with built-in sample images"""
    
    while True:
        print("\n" + "="*60)
        print("GRAYSCALE CONVERSION DEMO")
        print("="*60)
        print("1. Upload and convert your own image")
        print("2. Test with sample gradient image")
        print("3. Test with sample color pattern")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            load_and_convert_image()
        
        elif choice == '2':
            # Create a sample gradient image
            gradient = np.zeros((200, 300, 3), dtype=np.uint8)
            for i in range(200):
                val = int(i * 255 / 199)
                gradient[i, :, 0] = val  # Red gradient
                gradient[i, :, 1] = val  # Green gradient
                gradient[i, :, 2] = val  # Blue gradient
            
            result = convert_to_grayscale(gradient)
            
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].imshow(gradient)
            axes[0].set_title('Original Color Gradient')
            axes[0].axis('off')
            
            axes[1].imshow(result, cmap='gray')
            axes[1].set_title('Grayscale Result')
            axes[1].axis('off')
            
            plt.suptitle('Gradient Test Pattern')
            plt.show()
        
        elif choice == '3':
            # Create a color pattern
            pattern = np.zeros((300, 400, 3), dtype=np.uint8)
            
            # Red square (top-left)
            pattern[0:100, 0:100] = [255, 0, 0]
            
            # Green square (top-right)
            pattern[0:100, 300:400] = [0, 255, 0]
            
            # Blue square (bottom-left)
            pattern[200:300, 0:100] = [0, 0, 255]
            
            # White square (bottom-right)
            pattern[200:300, 300:400] = [255, 255, 255]
            
            # Yellow square (center)
            pattern[100:200, 150:250] = [255, 255, 0]
            
            result = convert_to_grayscale(pattern)
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(pattern)
            axes[0].set_title('Original Color Pattern')
            
            # Add labels for each color block
            axes[0].text(50, 50, f'Red\nRGB(255,0,0)', ha='center', va='center', 
                       color='white', fontsize=8, fontweight='bold')
            axes[0].text(350, 50, f'Green\nRGB(0,255,0)', ha='center', va='center', 
                       color='white', fontsize=8, fontweight='bold')
            axes[0].text(50, 250, f'Blue\nRGB(0,0,255)', ha='center', va='center', 
                       color='white', fontsize=8, fontweight='bold')
            axes[0].text(350, 250, f'White\nRGB(255,255,255)', ha='center', va='center', 
                       color='black', fontsize=8, fontweight='bold')
            axes[0].text(200, 150, f'Yellow\nRGB(255,255,0)', ha='center', va='center', 
                       color='black', fontsize=8, fontweight='bold')
            axes[0].axis('off')
            
            axes[1].imshow(result, cmap='gray')
            axes[1].set_title('Grayscale Result')
            
            # Add computed values for each block
            red_gray = int(0.299 * 255)
            green_gray = int(0.587 * 255)
            blue_gray = int(0.114 * 255)
            white_gray = 255
            yellow_gray = int(0.299 * 255 + 0.587 * 255)
            
            axes[1].text(50, 50, f'Red→{red_gray}', ha='center', va='center', 
                       color='white', fontsize=8, fontweight='bold')
            axes[1].text(350, 50, f'Green→{green_gray}', ha='center', va='center', 
                       color='black', fontsize=8, fontweight='bold')
            axes[1].text(50, 250, f'Blue→{blue_gray}', ha='center', va='center', 
                       color='white', fontsize=8, fontweight='bold')
            axes[1].text(350, 250, f'White→{white_gray}', ha='center', va='center', 
                       color='black', fontsize=8, fontweight='bold')
            axes[1].text(200, 150, f'Yellow→{yellow_gray}', ha='center', va='center', 
                       color='black', fontsize=8, fontweight='bold')
            axes[1].axis('off')
            
            plt.suptitle('Color to Grayscale Conversion Demo')
            plt.tight_layout()
            plt.show()
            
            print("\nGrayscale values for each color:")
            print(f"  Red (255,0,0)    → {red_gray}")
            print(f"  Green (0,255,0)  → {green_gray}")
            print(f"  Blue (0,0,255)   → {blue_gray}")
            print(f"  White (255,255,255) → {white_gray}")
            print(f"  Yellow (255,255,0)  → {yellow_gray}")
        
        elif choice == '4':
            print("Goodbye!")
            break
        
        else:
            print("Invalid option. Please try again.")


if __name__ == '__main__':
    demo_with_sample_images()