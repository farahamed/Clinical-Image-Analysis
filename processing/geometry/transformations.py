import numpy as np 
from processing.interpolation.zoom import zoom_image, bilinear_interpolation_pixel

def rotate(image, angle_degrees):
    # Convert angle to radians because sin & cos uses it
    angle_radian=np.radians(angle_degrees)
    #extract dimensions of image 
    h,w=image.shape[:2]
    #center of original image (rotattion is around center not corner)
    cx, cy= w/2, h/2
    #uses abs to ensure values are +ve
    cos_a=abs(np.cos(angle_radian))
    sin_a=abs(np.sin(angle_radian))
    #computes bounding  box bec rotated image may extend beyond original size
    new_w= int(h*sin_a + w*cos_a)
    new_h= int(h*cos_a + w*sin_a)
   
    #center of new image
    new_cx, new_cy= new_w/2, new_h/2
    #creat
    # ion of output image
    # check if image is RGB
    if image.ndim == 3:
        output= np.zeros((new_h, new_w, image.shape[2]), dtype=np.uint8) #creates empty RGB image
    else:
        output= np.zeros((new_h, new_w), dtype=np.uint8) #creates empty grayscale image

    #loop over every pizel in output, building image pixel by pixel (inverse mapping)
    for y_out in range(new_h):
        for x_out in range(new_w):
            #move origin to center, rotation assumes origin ar (0,0)
            dx, dy= x_out - new_cx, y_out - new_cy
            #apply inverse rotation to find corresponding source pixel in original image
            x_src= dx * np.cos(angle_radian) + dy * np.sin(angle_radian) + cx
            y_src= -dx * np.sin(angle_radian) + dy * np.cos(angle_radian) + cy
            #interpolation, get pixel from non integer location, clamp value between 0 & 255, preventing overflow
            output[y_out, x_out]=np.clip(bilinear_interpolation_pixel(image, x_src, y_src),0,255)
    return output


def shear(image, shear_x=0.0, shear_y=0.0):
    h, w = image.shape[:2]

    # Shear around the image center — mirrors how rotate() uses new_cx / new_cy
    cx, cy = w / 2, h / 2

    # Output stays the same size as input (same as rotate's new_w x new_h pattern
    # but here we keep original dims since shear is anchored to center)
    if image.ndim == 3:
        output = np.zeros((h, w, image.shape[2]), dtype=np.uint8)
    else:
        output = np.zeros((h, w), dtype=np.uint8)

    # Determinant of the shear matrix for exact inverse
    det = 1.0 - shear_x * shear_y

    for y_out in range(h):
        for x_out in range(w):
            # Translate to center origin (same trick as rotate's dx, dy)
            dx = x_out - cx
            dy = y_out - cy

            # Exact inverse shear, then translate back
            x_src = (dx - shear_x * dy) / det + cx
            y_src = (dy - shear_y * dx) / det + cy

            output[y_out, x_out] = np.clip(
                bilinear_interpolation_pixel(image, x_src, y_src), 0, 255
            )

    return output

    return output
#warpper function for zooming
def scale(image, scale_factor, method="bilinear"):
    return zoom_image(image, scale_factor, method)