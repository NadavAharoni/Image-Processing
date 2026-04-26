import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import correlate2d


def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: could not load image from '{path}'")
        sys.exit(1)
    return image


def get_sobel_kernels():
    sobel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]], dtype=np.float32)

    sobel_y = np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]], dtype=np.float32)
    return sobel_x, sobel_y


def apply_kernel(image, kernel):
    return correlate2d(image.astype(np.float32), kernel, mode='same')


def compute_magnitude(gx, gy):
    return np.sqrt(gx**2 + gy**2)


def display_results(image, gx, gy, magnitude):
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    images = [image, gx, gy, magnitude]
    titles = ['Original', 'Sobel X (Gx)', 'Sobel Y (Gy)', 'Magnitude']

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def normalize_to_uint8(arr):
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)


def save_results(input_path, image, gx, gy, magnitude):
    import os
    base, ext = os.path.splitext(input_path)
    cv2.imwrite(f"{base}_grayscale{ext}", image)
    cv2.imwrite(f"{base}_gx{ext}",        normalize_to_uint8(np.abs(gx)))
    cv2.imwrite(f"{base}_gy{ext}",        normalize_to_uint8(np.abs(gy)))
    cv2.imwrite(f"{base}_magnitude{ext}", normalize_to_uint8(magnitude))


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <image_path>")
        sys.exit(1)

    image = load_image(sys.argv[1])
    sobel_x, sobel_y = get_sobel_kernels()

    gx = apply_kernel(image, sobel_x)
    gy = apply_kernel(image, sobel_y)
    magnitude = compute_magnitude(gx, gy)

    save_results(sys.argv[1], image, gx, gy, magnitude)
    # display_results(image, gx, gy, magnitude)


if __name__ == "__main__":
    main()
