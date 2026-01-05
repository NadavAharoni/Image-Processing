import sys, time
import cv2, numpy as np
from PIL import Image, ImageChops

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} image-filename")
        exit(-1)

    filename = sys.argv[1]

    img = Image.open(filename)

    # PIL + numpy
    t0 = time.time()
    arr = np.array(img)
    arr += 50
    print("PIL + NumPy:", time.time() - t0, "sec")

    t0 = time.time()
    # cv2.add
    arr = np.array(img)
    cv2.add(arr, 50) # type: ignore
    print("PIL + cv2.add:", time.time() - t0, "sec")

    # PIL with ImageChops
    t0 = time.time()
    # Create a second image of the same size, filled with the constant value
    constant_img = Image.new(img.mode, img.size, color=(50, 50, 50))

    # Add the two images together, pixel by pixel
    # Pillow's add function handles overflow by wrapping around
    new_img = ImageChops.add(img, constant_img)
    print("PIL + ImageChops:", time.time() - t0, "sec")

    # ==
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    if img is None:
        print(f"failed to read image from {filename}")
        exit(-2)

    t0 = time.time()
    # cv2.add
    cv2.add(img, 50) # type: ignore
    print("cv2.add:", time.time() - t0, "sec")

    # numpy.add
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    if img is None:
        print(f"failed to read image from {filename}")
        exit(-2)

    t0 = time.time()
    np.add(img, 50)
    print("numpy.add:", time.time() - t0, "sec")

if __name__ == "__main__":
    main()



