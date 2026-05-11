import numpy as np


def convert_to_grayscale(image):
    """
    Converts an image to grayscale if needed.
    The project usually loads images as grayscale already, but this keeps the function safe.
    """
    if image is None:
        raise ValueError("No image provided.")

    if len(image.shape) == 2:  # 2D Image (height, width) Already grayscale
        return image.astype(np.uint8)

    if len(image.shape) == 3: # 3D Image (height, width, RGB color channels) Color image, convert to grayscale
        height, width, channels = image.shape
        gray = np.zeros((height, width), dtype=np.uint8)

        for i in range(height): # Rows
            for j in range(width): # Columns
                if channels >= 3:
                    r = image[i, j, 0]
                    g = image[i, j, 1]
                    b = image[i, j, 2]
                    gray[i, j] = int(0.299 * r + 0.587 * g + 0.114 * b) # Standard luminosity method
                else: # Handling Single-Channel 3D Images (DICOM)
                    gray[i, j] = image[i, j, 0]

        return gray

    raise ValueError("Unsupported image shape.")


def global_threshold(image, threshold_value):
    """
    Converts a grayscale image into a binary image using a global threshold.

    Pixels >= threshold become 255 (white).
    Pixels < threshold become 0 (black).
    """
    gray = convert_to_grayscale(image)

    height, width = gray.shape
    binary = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            if gray[i, j] >= threshold_value:
                binary[i, j] = 255
            else:
                binary[i, j] = 0

    return binary


def create_structuring_element(size, shape): # shape: determines the pattern of 1s and 0s inside the element during morphological operations
    """
    Creates a binary structuring element from scratch.

    Supported shapes:
    - Square
    - Cross
    """
    if size < 3:
        raise ValueError("Structuring element size must be at least 3.")

    if size % 2 == 0:
        raise ValueError("Structuring element size must be odd.")

    se = np.zeros((size, size), dtype=np.uint8)

    shape = shape.lower() # Convert shape to lowercase for case-insensitive comparison

    if shape == "square":
        for i in range(size):
            for j in range(size):
                se[i, j] = 1 # All pixels in the square structuring element are set to 1 (active)
                             # During morphological operations, ALL neighboring pixels are considered


    elif shape == "cross":
        center = size // 2 # Calculate the center pixel  of the structuring element

        for i in range(size):
            se[i, center] = 1 # Set the center row to 1 (active)
                              # During morphological operations, these pixels are considered

        for j in range(size):
            se[center, j] = 1 # Set the center column to 1 (active)
                               # During morphological operations, these pixels are considered

    else:
        raise ValueError("Unsupported structuring element shape.")

    return se


def ensure_binary(image):
    """
    Makes sure the input is treated as a binary mask.
    Any pixel > 0 becomes 255.
    Any pixel == 0 stays 0.
    """
    gray = convert_to_grayscale(image)

    height, width = gray.shape
    binary = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            if gray[i, j] > 0:
                binary[i, j] = 255
            else:
                binary[i, j] = 0

    return binary


def pad_binary_image(binary_image, pad_h, pad_w):
    """
    Pads a binary image with zeros manually.
    This handles borders for erosion and dilation.
    """
    height, width = binary_image.shape

    padded_height = height + 2 * pad_h
    padded_width = width + 2 * pad_w

    padded = np.zeros((padded_height, padded_width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            padded[i + pad_h, j + pad_w] = binary_image[i, j]

    return padded


def erode(binary_image, structuring_element):
    """
    Erosion from scratch.

    A pixel remains white only if all active SE positions fit completely
    inside the white foreground region.
    """
    binary = ensure_binary(binary_image)

    image_h, image_w = binary.shape
    se_h, se_w = structuring_element.shape

    pad_h = se_h // 2
    pad_w = se_w // 2

    padded = pad_binary_image(binary, pad_h, pad_w)
    output = np.zeros((image_h, image_w), dtype=np.uint8)

    for i in range(image_h):
        for j in range(image_w):
            fits = True

            for m in range(se_h):
                for n in range(se_w):
                    if structuring_element[m, n] == 1: # Only check active SE positions
                        if padded[i + m, j + n] != 255:
                            fits = False
                            break

                if not fits:
                    break

            if fits:
                output[i, j] = 255
            else:
                output[i, j] = 0

    return output


def dilate(binary_image, structuring_element):
    """
    Dilation from scratch.

    A pixel becomes white if at least one active SE position overlaps
    with a white foreground pixel.
    """
    binary = ensure_binary(binary_image)

    image_h, image_w = binary.shape
    se_h, se_w = structuring_element.shape

    pad_h = se_h // 2
    pad_w = se_w // 2

    padded = pad_binary_image(binary, pad_h, pad_w)
    output = np.zeros((image_h, image_w), dtype=np.uint8)

    for i in range(image_h):
        for j in range(image_w):
            overlaps = False

            for m in range(se_h):
                for n in range(se_w):
                    if structuring_element[m, n] == 1: # Only check active SE positions
                        if padded[i + m, j + n] == 255:
                            overlaps = True
                            break

                if overlaps:
                    break

            if overlaps:
                output[i, j] = 255
            else:
                output[i, j] = 0

    return output


def opening(binary_image, structuring_element):
    """
    Opening = Erosion followed by Dilation.
    Used to remove small white noise.
    """
    eroded = erode(binary_image, structuring_element)
    opened = dilate(eroded, structuring_element)
    return opened


def closing(binary_image, structuring_element):
    """
    Closing = Dilation followed by Erosion.
    Used to fill small holes and gaps.
    """
    dilated = dilate(binary_image, structuring_element)
    closed = erode(dilated, structuring_element)
    return closed


def boundary_extraction(binary_image, structuring_element):
    """
    Boundary Extraction = Original binary image - Eroded image.
    This outlines the borders of anatomical structures.
    """
    binary = ensure_binary(binary_image)
    eroded = erode(binary, structuring_element)

    height, width = binary.shape
    boundary = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            value = int(binary[i, j]) - int(eroded[i, j])

            if value < 0:
                value = 0

            boundary[i, j] = value

    return boundary