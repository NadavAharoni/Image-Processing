import sys
import os
import numpy as np
import cv2

def brighten(img, b, func):
    """
    Brighten an image by adding a value to all pixels.
    
    Args:
        img: Grayscale image
        b: Integer value to add to all pixels
        func: String - "np" for numpy.add or "cv2" for cv2.add
    
    Returns:
        Brightened image
    """
    if func == "np":
        return np.add(img, b)
    elif func == "cv2":
        return cv2.add(img, b)
    else:
        raise ValueError("func must be 'np' or 'cv2'")
    

def main():
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} image-filename brightness_change")
        exit(-1)

    file_name = sys.argv[1]
    brightness_change = float(sys.argv[2])
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)
    if img is None:
        print(f"failed to read image from {file_name}")
        exit(-2)
    
    brightened_img = brighten(img, brightness_change, "cv2")
    file, ext = os.path.splitext(file_name)
    cv2.imwrite(f'{file}_brightened_cv2_{brightness_change}.{ext}', brightened_img)

if __name__ == "__main__":
    main()
