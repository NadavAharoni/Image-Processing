import sys
import os
import numpy as np
import cv2

def brighten(img, b, func):
    """Brighten an image by adding a value to all pixels.

    This function is intended for educational purposes to demonstrate
    the difference between `np.add` (which will overflow/wrap for
    uint8 pixel arrays) and `cv2.add` (which saturates/clamps values).

    Args:
        img: numpy array image (grayscale, BGR, or BGRA)
        b: integer value to add to all pixels
        func: "np" to use `np.add` (demonstrates overflow) or "cv2" to use
              `cv2.add` (saturating behavior)

    Returns:
        Brightened image with same shape and dtype as input.
    """
    # ensure brightness is integer
    b = int(b)

    if img.ndim == 2:
        # single channel
        if func == "np":
            res = np.add(img, b)
        elif func == "cv2":
            res = cv2.add(img, b)
        else:
            raise ValueError("func must be 'np' or 'cv2'")
        return res

    # multi-channel: handle alpha separately if present
    if img.ndim == 3:
        channels = img.shape[2]
        if channels == 4:
            bgr = img[:, :, :3]
            alpha = img[:, :, 3]
            if func == "np":
                bgr_res = np.add(bgr, b)
            elif func == "cv2":
                bgr_res = cv2.add(bgr, b)
            else:
                raise ValueError("func must be 'np' or 'cv2'")
            return cv2.merge((bgr_res[:, :, 0], bgr_res[:, :, 1], bgr_res[:, :, 2], alpha))
        else:
            # 3-channel (BGR) or other multi-channel
            if func == "np":
                res = np.add(img, b)
            elif func == "cv2":
                res = cv2.add(img, b)
            else:
                raise ValueError("func must be 'np' or 'cv2'")
            return res

    raise ValueError("Unsupported image shape for brighten()")
    

def main():
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} image-filename brightness_change")
        exit(-1)

    file_name = sys.argv[1]
    try:
        brightness_change = int(sys.argv[2])
    except ValueError:
        print(f"Error: brightness_change must be an integer, got '{sys.argv[2]}'")
        exit(-3)
    img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"failed to read image from {file_name}")
        exit(-2)
    
    brightened_img = brighten(img, brightness_change, "cv2")
    file, ext = os.path.splitext(file_name)
    cv2.imwrite(f'{file}_brightened_cv2_{brightness_change}{ext}', brightened_img)

    brightened_img = brighten(img, brightness_change, "np")
    file, ext = os.path.splitext(file_name)
    cv2.imwrite(f'{file}_brightened_np_{brightness_change}{ext}', brightened_img)

if __name__ == "__main__":
    main()
