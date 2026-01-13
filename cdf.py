import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Visualize image, histogram, and CDF with fixed scaling.")
    parser.add_argument("input_file", help="Path to the input image file")
    args = parser.parse_args()

    # Load image (preserve color channels if present)
    img = cv2.imread(args.input_file, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Could not open image '{args.input_file}'")
        return
    # If image has alpha channel, convert to BGR
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def hist_and_cdf(channel):
        hist, bins = np.histogram(channel.flatten(), bins=256, range=[0, 256])
        cdf = hist.cumsum()
        cdf_norm = cdf / cdf.max() if cdf.max() > 0 else cdf
        return hist, cdf_norm

    def plot_hist(ax, hist, color='gray'):
        ax.bar(range(256), hist, color=color, width=1.0)
        ax.set_xlim([0, 256])

    # Handle grayscale image
    if len(img.shape) == 2:
        hist, cdf_norm = hist_and_cdf(img)
        # Equalize
        img_eq = cv2.equalizeHist(img)
        hist_eq, cdf_eq = hist_and_cdf(img_eq)

        fig, axs = plt.subplots(1, 6, figsize=(18, 4))

        axs[0].imshow(img, cmap='gray', vmin=0, vmax=255)
        axs[0].set_title('Original')
        axs[0].axis('off')

        axs[1].bar(range(256), hist, color='gray', width=1.0)
        axs[1].set_title('Histogram')
        axs[1].set_xlim([0, 256])

        axs[2].plot(range(256), cdf_norm, color='blue')
        axs[2].set_title('CDF')
        axs[2].set_xlim([0, 256])
        axs[2].set_ylim([0, 1])

        axs[3].bar(range(256), hist_eq, color='gray', width=1.0)
        axs[3].set_title('Eq Histogram')
        axs[3].set_xlim([0, 256])

        axs[4].plot(range(256), cdf_eq, color='blue')
        axs[4].set_title('Eq CDF')
        axs[4].set_xlim([0, 256])
        axs[4].set_ylim([0, 1])

        axs[5].imshow(img_eq, cmap='gray', vmin=0, vmax=255)
        axs[5].set_title('Equalized')
        axs[5].axis('off')

        plt.tight_layout()
        plt.show()
        return

    # Color image handling: show per-channel rows plus a full-color row
    # Convert to RGB for plotting convenience
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    channels = {'R': img_rgb[..., 0], 'G': img_rgb[..., 1], 'B': img_rgb[..., 2]}

    n_rows = 4  # R, G, B, and full-color
    n_cols = 6  # original channel image, hist, cdf, eq hist, eq cdf, result image

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))

    # For each channel row
    for i, (ch_name, ch_data) in enumerate(channels.items()):
        # Original channel as colored image (show only that channel)
        ch_img_color = np.zeros_like(img_rgb)
        if ch_name == 'R':
            ch_img_color[..., 0] = ch_data
        elif ch_name == 'G':
            ch_img_color[..., 1] = ch_data
        else:
            ch_img_color[..., 2] = ch_data

        hist, cdf_norm = hist_and_cdf(ch_data)
        ch_eq = cv2.equalizeHist(ch_data)
        hist_eq, cdf_eq = hist_and_cdf(ch_eq)

        # Resulting channel image in color
        ch_img_eq_color = np.zeros_like(img_rgb)
        if ch_name == 'R':
            ch_img_eq_color[..., 0] = ch_eq
        elif ch_name == 'G':
            ch_img_eq_color[..., 1] = ch_eq
        else:
            ch_img_eq_color[..., 2] = ch_eq

        ax_img = axes[i, 0]
        ax_img.imshow(ch_img_color)
        ax_img.set_title(f'{ch_name} (orig)')
        ax_img.axis('off')

        ax_hist = axes[i, 1]
        ax_hist.bar(range(256), hist.flatten(), color=ch_name.lower(), width=1.0)
        ax_hist.set_xlim([0, 256])
        ax_hist.set_title('Histogram')

        ax_cdf = axes[i, 2]
        ax_cdf.plot(range(256), cdf_norm, color=ch_name.lower())
        ax_cdf.set_xlim([0, 256])
        ax_cdf.set_ylim([0, 1])
        ax_cdf.set_title('CDF')

        ax_hist_eq = axes[i, 3]
        ax_hist_eq.bar(range(256), hist_eq.flatten(), color=ch_name.lower(), width=1.0)
        ax_hist_eq.set_xlim([0, 256])
        ax_hist_eq.set_title('Eq Histogram')

        ax_cdf_eq = axes[i, 4]
        ax_cdf_eq.plot(range(256), cdf_eq, color=ch_name.lower())
        ax_cdf_eq.set_xlim([0, 256])
        ax_cdf_eq.set_ylim([0, 1])
        ax_cdf_eq.set_title('Eq CDF')

        ax_res = axes[i, 5]
        ax_res.imshow(ch_img_eq_color)
        ax_res.set_title(f'{ch_name} (equalized)')
        ax_res.axis('off')

    # Final row: full color image and full histograms (3 channels)
    row = 3
    axes[row, 0].imshow(img_rgb)
    axes[row, 0].set_title('Original RGB')
    axes[row, 0].axis('off')

    # Histogram: plot 3 channel histograms
    for ci, color in enumerate(('r', 'g', 'b')):
        hist = cv2.calcHist([img_rgb], [ci], None, [256], [0, 256]).flatten()
        axes[row, 1].plot(hist, color=color)
    axes[row, 1].set_xlim([0, 256])
    axes[row, 1].set_title('RGB Histograms')

    # CDFs
    for ci, color in enumerate(('r', 'g', 'b')):
        hist = cv2.calcHist([img_rgb], [ci], None, [256], [0, 256]).flatten()
        cdf = hist.cumsum()
        cdf = cdf / cdf.max() if cdf.max() > 0 else cdf
        axes[row, 2].plot(cdf, color=color)
    axes[row, 2].set_xlim([0, 256])
    axes[row, 2].set_ylim([0, 1])
    axes[row, 2].set_title('RGB CDFs')

    # Equalize each channel and show hist/cdf
    img_eq_rgb = img_rgb.copy()
    for ci in range(3):
        img_eq_rgb[..., ci] = cv2.equalizeHist(img_eq_rgb[..., ci])

    for ci, color in enumerate(('r', 'g', 'b')):
        hist_eq = cv2.calcHist([img_eq_rgb], [ci], None, [256], [0, 256]).flatten()
        axes[row, 3].plot(hist_eq, color=color)
    axes[row, 3].set_xlim([0, 256])
    axes[row, 3].set_title('Eq RGB Hist')

    for ci, color in enumerate(('r', 'g', 'b')):
        hist_eq = cv2.calcHist([img_eq_rgb], [ci], None, [256], [0, 256]).flatten()
        cdf_eq = hist_eq.cumsum()
        cdf_eq = cdf_eq / cdf_eq.max() if cdf_eq.max() > 0 else cdf_eq
        axes[row, 4].plot(cdf_eq, color=color)
    axes[row, 4].set_xlim([0, 256])
    axes[row, 4].set_ylim([0, 1])
    axes[row, 4].set_title('Eq RGB CDF')

    axes[row, 5].imshow(img_eq_rgb)
    axes[row, 5].set_title('Equalized RGB (per-channel)')
    axes[row, 5].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
