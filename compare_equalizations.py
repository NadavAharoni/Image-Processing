import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(description="Visualize image, histogram, and CDF with fixed scaling.")
    parser.add_argument("input_file", help="Path to the input image file")
    args = parser.parse_args()

    # --------------------------------------------------
    # Read image and convert to grayscale
    # --------------------------------------------------
    img = cv2.imread(args.input_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --------------------------------------------------
    # 1. Min–max normalization
    # --------------------------------------------------
    minmax = cv2.normalize(
        gray, None,
        alpha=0, beta=255,
        norm_type=cv2.NORM_MINMAX
    )

    # --------------------------------------------------
    # 2. Global histogram equalization
    # --------------------------------------------------
    histeq = cv2.equalizeHist(gray)

    # --------------------------------------------------
    # 3. Histogram equalization with dithering
    # --------------------------------------------------
    noise_amplitude = 2  # very small on purpose
    noise = np.random.randint(
        -noise_amplitude,
        noise_amplitude + 1,
        size=gray.shape,
        dtype=np.int16
    )

    gray_dithered = np.clip(
        gray.astype(np.int16) + noise,
        0, 255
    ).astype(np.uint8)

    histeq_dithered = cv2.equalizeHist(gray_dithered)

    # --------------------------------------------------
    # 4. CLAHE
    # --------------------------------------------------
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    clahe_img = clahe.apply(gray)

    # --------------------------------------------------
    # Display results
    # --------------------------------------------------
    titles = [
        "Original",
        "Min–max normalization",
        "Histogram equalization",
        "Hist. eq. + dithering",
        "CLAHE"
    ]

    images = [
        gray,
        minmax,
        histeq,
        histeq_dithered,
        clahe_img
    ]

    plt.figure(figsize=(14, 4))
    for i, (title, im) in enumerate(zip(titles, images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(im, cmap="gray", vmin=0, vmax=255)
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()