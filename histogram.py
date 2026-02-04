import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_and_display_histogram(image_path):
    """
    Calculates the histogram of an image using cv2 and displays it using matplotlib.

    Args:
        image_path (str): The path to the image file.
    """
    # 1. Load the image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Convert BGR to RGB for correct matplotlib display if showing the image
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    # 2. Determine if the image is color or grayscale
    if len(img.shape) == 3:
        # Color image: calculate histogram for each channel
        colors = ('b', 'g', 'r')
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), vmin=0, vmax=255) # Display original image
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        for i, color in enumerate(colors):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(hist, color=color)
            plt.xlim([0, 256])
        # set x-axis labels every 25 for consistency with grayscale histogram
        ax2 = plt.gca()
        xticks = np.arange(0, 256, 25)
        ax2.set_xticks(xticks)
        ax2.set_xticklabels([str(t) for t in xticks])
        
        plt.title('Color Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
    else:
        # Grayscale image: calculate single histogram
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        print(f"hist.shape={hist.shape}")
        print(hist[100:110,:])

        fig, axs = plt.subplots(1, 2, figsize=(12, 5),
                                gridspec_kw={'width_ratios': [1, 2]})
        axs[0].imshow(img, cmap='gray', vmin=0, vmax=255)
        axs[0].set_title('Original Grayscale Image')
        axs[0].axis('off')


        x = np.arange(256)
        h_flat = hist.flatten()
        bar_colors = ['#bbbbbb' if i % 2 == 0 else '#666666' for i in range(256)]
        axs[1].bar(x, h_flat, width=1.0, color=bar_colors, edgecolor='black', linewidth=0.3)
        # set x-axis labels
        xticks = np.arange(0, 256, 25)
        axs[1].set_xticks(xticks)
        axs[1].set_xticklabels([str(t) for t in xticks])
        axs[1].set_xlim([0, 256])
        axs[1].set_title('Grayscale Histogram')
        axs[1].set_xlabel('Pixel Value')
        axs[1].set_ylabel('Frequency')
        plt.tight_layout()

    # 3. Display the plots
    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} image-filename")
        exit(-1)

    file_name = sys.argv[1]

    # Example with a color image:
    calculate_and_display_histogram(file_name)

    # Example with a grayscale image (if you have one):
    # calculate_and_display_histogram('sample_gray_image.jpg')
    

# --- Example Usage ---
# Ensure you have an image named 'sample_image.jpg' in your directory
# or replace 'sample_image.jpg' with the path to your image.
if __name__ == '__main__':
    main()
