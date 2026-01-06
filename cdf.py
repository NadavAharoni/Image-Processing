import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Visualize image, histogram, and CDF with fixed scaling.")
    parser.add_argument("input_file", help="Path to the input image file")
    args = parser.parse_args()

    # Load image in grayscale
    img = cv2.imread(args.input_file, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not open image '{args.input_file}'")
        return

    # Calculate Histogram and CDF
    hist, bins = np.histogram(img.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf.max()

    # Create the display
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Grayscale Image with fixed scaling
    # vmin and vmax ensure 0 is black and 255 is white regardless of image content
    ax1.imshow(img, cmap='gray', vmin=0, vmax=255)
    ax1.set_title('Grayscale Image (0-255)')
    ax1.axis('off')

    # Plot 2: Histogram (Pixel Counts)
    ax2.bar(range(256), hist, color='gray', width=1.0)
    ax2.set_title('Histogram')
    ax2.set_xlim([0, 256])

    # Plot 3: Cumulative Distribution Function (CDF)
    ax3.plot(range(256), cdf_normalized, color='blue')
    ax3.set_title('Normalized CDF')
    ax3.set_xlim([0, 256])
    ax3.set_ylim([0, 1])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
