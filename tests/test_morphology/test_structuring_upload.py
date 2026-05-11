import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
import os

# Add the project root to the Python path to import from processing module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the function from your existing module
from processing.morphology.binary_morphology import create_structuring_element


def visualize_structuring_element(se, title=None):
    """Visualize a structuring element with a nice grid display"""
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Display as image
    im = axes[0].imshow(se, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f'{title}\n(Image View)', fontsize=12, fontweight='bold')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    # Add grid lines
    for i in range(se.shape[0] + 1):
        axes[0].axhline(y=i - 0.5, color='blue', linewidth=1, alpha=0.5)
        axes[0].axvline(x=i - 0.5, color='blue', linewidth=1, alpha=0.5)
    
    # Display as matrix with values
    axes[1].set_title(f'{title}\n(Matrix View - 1=Foreground, 0=Background)', fontsize=12, fontweight='bold')
    
    # Create a colored table
    table_data = []
    cell_colors = []
    for i in range(se.shape[0]):
        row_data = []
        row_colors = []
        for j in range(se.shape[1]):
            val = se[i, j]
            if val == 1:
                row_data.append('1')
                row_colors.append('lightgreen')
            else:
                row_data.append('0')
                row_colors.append('lightcoral')
        table_data.append(row_data)
        cell_colors.append(row_colors)
    
    # Create table
    table = axes[1].table(
        cellText=table_data,
        cellColours=cell_colors,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Make cells square-ish
    for key, cell in table.get_celld().items():
        cell.set_height(0.1)
        cell.set_width(0.1)
    
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


def interactive_se_creator():
    """Interactive structuring element creator and visualizer"""
    
    while True:
        print("\n" + "="*60)
        print("STRUCTURING ELEMENT VISUALIZER")
        print("Using: processing.morphology.binary_morphology.create_structuring_element")
        print("="*60)
        print("1. Create and visualize a single element")
        print("2. Compare square elements of different sizes")
        print("3. Compare cross elements of different sizes")
        print("4. Side-by-side comparison (square vs cross)")
        print("5. Test invalid inputs (error handling demo)")
        print("6. Show function source code")
        print("7. Exit")
        
        choice = input("\nSelect option (1-7): ").strip()
        
        if choice == '1':
            # Single element creation
            print("\n" + "-"*40)
            print("CREATE SINGLE STRUCTURING ELEMENT")
            print("-"*40)
            
            # Get shape type
            while True:
                shape = input("Enter shape (square/cross): ").strip().lower()
                if shape in ['square', 'cross']:
                    break
                print("Invalid shape. Choose 'square' or 'cross'")
            
            # Get size
            while True:
                try:
                    size = int(input("Enter size (odd number ≥ 3): ").strip())
                    if size >= 3 and size % 2 == 1:
                        break
                    else:
                        print("Size must be an odd number ≥ 3")
                except ValueError:
                    print("Please enter a valid number")
            
            # Create and visualize
            try:
                se = create_structuring_element(size, shape)
                print(f"\nCreated {size}x{size} {shape} structuring element")
                print(f"Shape: {se.shape}")
                print(f"Data type: {se.dtype}")
                print(f"Number of 1s (foreground): {np.sum(se == 1)}")
                print(f"Number of 0s (background): {np.sum(se == 0)}")
                
                # Print matrix
                print("\nMatrix:")
                for row in se:
                    print("  ", " ".join(str(x) for x in row))
                
                visualize_structuring_element(se, f"{size}x{size} {shape.capitalize()}")
                
            except ValueError as e:
                print(f"Error: {e}")
        
        elif choice == '2':
            # Compare square elements
            print("\n" + "-"*40)
            print("COMPARE SQUARE ELEMENTS")
            print("-"*40)
            
            sizes = []
            for i in range(3):
                while True:
                    try:
                        size = int(input(f"Enter size {i+1} (odd number ≥ 3): ").strip())
                        if size >= 3 and size % 2 == 1:
                            sizes.append(size)
                            break
                        else:
                            print("Size must be an odd number ≥ 3")
                    except ValueError:
                        print("Please enter a valid number")
            
            # Create and display
            fig, axes = plt.subplots(1, len(sizes), figsize=(5*len(sizes), 5))
            if len(sizes) == 1:
                axes = [axes]
            
            for idx, size in enumerate(sizes):
                se = create_structuring_element(size, "square")
                axes[idx].imshow(se, cmap='gray', vmin=0, vmax=1)
                axes[idx].set_title(f'{size}x{size} Square\n1s: {np.sum(se == 1)}', fontsize=12)
                axes[idx].set_xticks([])
                axes[idx].set_yticks([])
                
                # Add grid
                for i in range(se.shape[0] + 1):
                    axes[idx].axhline(y=i - 0.5, color='blue', linewidth=1, alpha=0.5)
                    axes[idx].axvline(x=i - 0.5, color='blue', linewidth=1, alpha=0.5)
            
            plt.suptitle('Square Structuring Elements Comparison', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            # Print matrices
            print("\nMatrices:")
            for size in sizes:
                se = create_structuring_element(size, "square")
                print(f"\n{size}x{size} Square:")
                for row in se:
                    print("  ", " ".join(str(x) for x in row))
        
        elif choice == '3':
            # Compare cross elements
            print("\n" + "-"*40)
            print("COMPARE CROSS ELEMENTS")
            print("-"*40)
            
            sizes = []
            for i in range(3):
                while True:
                    try:
                        size = int(input(f"Enter size {i+1} (odd number ≥ 3): ").strip())
                        if size >= 3 and size % 2 == 1:
                            sizes.append(size)
                            break
                        else:
                            print("Size must be an odd number ≥ 3")
                    except ValueError:
                        print("Please enter a valid number")
            
            # Create and display
            fig, axes = plt.subplots(1, len(sizes), figsize=(5*len(sizes), 5))
            if len(sizes) == 1:
                axes = [axes]
            
            for idx, size in enumerate(sizes):
                se = create_structuring_element(size, "cross")
                axes[idx].imshow(se, cmap='gray', vmin=0, vmax=1)
                axes[idx].set_title(f'{size}x{size} Cross\n1s: {np.sum(se == 1)}', fontsize=12)
                axes[idx].set_xticks([])
                axes[idx].set_yticks([])
                
                # Add grid
                for i in range(se.shape[0] + 1):
                    axes[idx].axhline(y=i - 0.5, color='blue', linewidth=1, alpha=0.5)
                    axes[idx].axvline(x=i - 0.5, color='blue', linewidth=1, alpha=0.5)
            
            plt.suptitle('Cross Structuring Elements Comparison', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            # Print matrices
            print("\nMatrices:")
            for size in sizes:
                se = create_structuring_element(size, "cross")
                print(f"\n{size}x{size} Cross:")
                for row in se:
                    print("  ", " ".join(str(x) for x in row))
        
        elif choice == '4':
            # Side-by-side comparison
            print("\n" + "-"*40)
            print("SQUARE VS CROSS COMPARISON")
            print("-"*40)
            
            while True:
                try:
                    size = int(input("Enter size (odd number ≥ 3): ").strip())
                    if size >= 3 and size % 2 == 1:
                        break
                    else:
                        print("Size must be an odd number ≥ 3")
                except ValueError:
                    print("Please enter a valid number")
            
            se_square = create_structuring_element(size, "square")
            se_cross = create_structuring_element(size, "cross")
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Square
            axes[0].imshow(se_square, cmap='gray', vmin=0, vmax=1)
            axes[0].set_title(f'{size}x{size} Square\nForeground: {np.sum(se_square == 1)} | Background: {np.sum(se_square == 0)}', 
                            fontsize=12, fontweight='bold')
            axes[0].set_xticks([])
            axes[0].set_yticks([])
            for i in range(se_square.shape[0] + 1):
                axes[0].axhline(y=i - 0.5, color='blue', linewidth=1, alpha=0.5)
                axes[0].axvline(x=i - 0.5, color='blue', linewidth=1, alpha=0.5)
            
            # Cross
            axes[1].imshow(se_cross, cmap='gray', vmin=0, vmax=1)
            axes[1].set_title(f'{size}x{size} Cross\nForeground: {np.sum(se_cross == 1)} | Background: {np.sum(se_cross == 0)}', 
                            fontsize=12, fontweight='bold')
            axes[1].set_xticks([])
            axes[1].set_yticks([])
            for i in range(se_cross.shape[0] + 1):
                axes[1].axhline(y=i - 0.5, color='blue', linewidth=1, alpha=0.5)
                axes[1].axvline(x=i - 0.5, color='blue', linewidth=1, alpha=0.5)
            
            plt.suptitle(f'Square vs Cross ({size}x{size})', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            # Print matrices side by side
            print(f"\n{size}x{size} Square:                    {size}x{size} Cross:")
            for row_sq, row_cr in zip(se_square, se_cross):
                print(f"  {' '.join(str(x) for x in row_sq)}           {' '.join(str(x) for x in row_cr)}")
        
        elif choice == '5':
            # Error handling demo
            print("\n" + "-"*40)
            print("ERROR HANDLING DEMO")
            print("-"*40)
            
            test_cases = [
                ("Size < 3", 1, "square"),
                ("Size = 2", 2, "cross"),
                ("Even size", 4, "square"),
                ("Invalid shape", 3, "circle"),
                ("Empty string shape", 3, ""),
            ]
            
            print("Testing various invalid inputs:\n")
            for desc, size, shape in test_cases:
                try:
                    print(f"Test: {desc} (size={size}, shape='{shape}')")
                    se = create_structuring_element(size, shape)
                    print(f"  ✗ Should have raised ValueError!")
                except ValueError as e:
                    print(f"  ✓ Correctly raised ValueError: {e}")
                except Exception as e:
                    print(f"  ⚠ Raised unexpected {type(e).__name__}: {e}")
                print()
        
        elif choice == '6':
            # Show source code
            import inspect
            print("\n" + "="*60)
            print("Function source code:")
            print("="*60)
            print(inspect.getsource(create_structuring_element))
            print("="*60)
            print(f"Function location: {inspect.getfile(create_structuring_element)}")
        
        elif choice == '7':
            print("Goodbye!")
            break
        
        else:
            print("Invalid option. Please try again.")


def quick_demo():
    """Quick demonstration of common structuring elements"""
    
    print("\n" + "="*60)
    print("QUICK DEMO - Common Structuring Elements")
    print("="*60)
    
    # Create common elements
    elements = [
        create_structuring_element(3, "square"),
        create_structuring_element(3, "cross"),
        create_structuring_element(5, "square"),
        create_structuring_element(5, "cross"),
        create_structuring_element(7, "cross"),
    ]
    titles = [
        "3x3 Square",
        "3x3 Cross", 
        "5x5 Square",
        "5x5 Cross",
        "7x7 Cross"
    ]
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    for idx, (se, title) in enumerate(zip(elements, titles)):
        axes[idx].imshow(se, cmap='gray', vmin=0, vmax=1)
        axes[idx].set_title(f'{title}\nForeground: {np.sum(se == 1)}', fontsize=10, fontweight='bold')
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
        
        # Add grid
        for i in range(se.shape[0] + 1):
            axes[idx].axhline(y=i - 0.5, color='blue', linewidth=0.5, alpha=0.5)
            axes[idx].axvline(x=i - 0.5, color='blue', linewidth=0.5, alpha=0.5)
    
    plt.suptitle('Common Structuring Elements Overview', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Print all matrices
    print("\nMatrices:")
    for title, se in zip(titles, elements):
        print(f"\n{title}:")
        for row in se:
            print("  ", " ".join(str(x) for x in row))


if __name__ == '__main__':
    # Verify the import works
    try:
        import inspect
        print(f"Successfully imported create_structuring_element from:")
        print(f"  {inspect.getfile(create_structuring_element)}")
        print(f"  Function signature: {inspect.signature(create_structuring_element)}")
    except ImportError as e:
        print(f"Error importing function: {e}")
        print("Make sure the processing module is in your Python path")
        sys.exit(1)
    
    # Show quick demo
    quick_demo()
    
    # Launch interactive mode
    interactive_se_creator()