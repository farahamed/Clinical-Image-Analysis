import os
import cv2
from processing.filtering.edge_detection import apply_edge_detection



BASE_DIR = os.path.abspath(os.path.join(__file__, "../../.."))
IMAGE_PATH = os.path.join(BASE_DIR, "assets", "test.jpeg")


image = cv2.imread(IMAGE_PATH, 0)

if image is None:
    raise FileNotFoundError(f"Could not load image at: {IMAGE_PATH}")

image = cv2.resize(image, (256, 256))

print("Running Sobel Edge Detection...")


gx, gy, mag = apply_edge_detection(image)


cv2.imshow("Original", image)
cv2.imshow("Gradient X", cv2.convertScaleAbs(gx))
cv2.imshow("Gradient Y", cv2.convertScaleAbs(gy))
cv2.imshow("Magnitude", cv2.convertScaleAbs(mag))

cv2.waitKey(0)
cv2.destroyAllWindows()