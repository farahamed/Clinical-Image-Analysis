import numpy as np
from PIL import Image
import pydicom


def load_regular_image(file_path):
    pil_image = Image.open(file_path)
    pil_image = pil_image.convert('L')
    image_array = np.array(pil_image)
    return image_array


def load_dicom_image(file_path):
    dicom_data = pydicom.dcmread(file_path)
    pixel_array = dicom_data.pixel_array

    pixel_min = pixel_array.min()
    pixel_max = pixel_array.max()

    if pixel_max != pixel_min:
        pixel_array = ((pixel_array - pixel_min) /
                       (pixel_max - pixel_min) * 255)
    pixel_array = pixel_array.astype(np.uint8)

    if len(pixel_array.shape) == 3:
        pixel_array = pixel_array[0]

    return pixel_array, dicom_data


def get_dicom_tag(dicom_data, tag_name):
    try:
        value = getattr(dicom_data, tag_name, "N/A")
        return str(value) if value != "N/A" else "N/A"
    except:
        return "N/A"