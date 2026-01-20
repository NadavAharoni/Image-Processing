import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse


def histogram_from_channel(channel):
    """Compute 256-bin histogram for a single-channel uint8 image using numpy."""
    flat = channel.flatten()
    hist = np.bincount(flat, minlength=256)
    return hist.astype(np.int32)


def cdf_from_hist(hist):
    """Compute cumulative distribution function (CDF) from histogram."""
    cdf = hist.cumsum().astype(np.float64)
    return cdf


def mapping_from_cdf(cdf):
    """Create a 256-entry mapping table from the CDF to [0..255].

    The mapping is: round(255 * (cdf - cdf_min) / (cdf_max - cdf_min)).
    Handles constant images by returning identity mapping.
    """
    cdf_min = cdf[cdf > 0].min() if np.any(cdf > 0) else 0.0
    cdf_max = cdf.max() if cdf.max() > 0 else 0.0

    if cdf_max == cdf_min:
        return np.arange(256, dtype=np.uint8)

    norm = (cdf - cdf_min) / (cdf_max - cdf_min)
    map_vals = np.floor(255.0 * norm + 0.5).clip(0, 255).astype(np.uint8)
    return map_vals


def equalize_channel_pixelwise(channel):
    """Equalize a single channel using pixelwise mapping (no cv2.equalizeHist)."""
    hist = histogram_from_channel(channel)
    cdf = cdf_from_hist(hist)
    mapping = mapping_from_cdf(cdf)
    # Apply mapping using vectorized indexing
    return mapping[channel]


def equalize_image_pixelwise(img, per_channel=True):
    """Equalize an image. If grayscale, operate directly.
    If color and per_channel True, equalize each channel separately.
    Returns image with same shape and dtype uint8.
    """
    if img.ndim == 2:
        return equalize_channel_pixelwise(img)

    # color image
    out = img.copy()
    if per_channel:
        for ch in range(out.shape[2]):
            out[..., ch] = equalize_channel_pixelwise(out[..., ch])
    else:
        # Flatten across channels: compute histogram of luminance (average) and map each channel
        lum = out.mean(axis=2).astype(np.uint8)
        hist = histogram_from_channel(lum)
        cdf = cdf_from_hist(hist)
        mapping = mapping_from_cdf(cdf)
        for ch in range(out.shape[2]):
            out[..., ch] = mapping[out[..., ch]]

    return out


def plot_side_by_side(orig, eq, title_prefix=''):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Images
    axes[0, 0].imshow(orig, vmin=0, vmax=255)
    axes[0, 0].set_title(f'{title_prefix}Original')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(eq, vmin=0, vmax=255)
    axes[0, 1].set_title(f'{title_prefix}Equalized')
    axes[0, 1].axis('off')

    # Histograms
    if orig.ndim == 2:
        hist_o = histogram_from_channel(orig)
        hist_e = histogram_from_channel(eq)
        axes[1, 0].bar(range(256), hist_o, color='gray')
        axes[1, 0].set_title('Original Histogram')
        axes[1, 1].bar(range(256), hist_e, color='gray')
        axes[1, 1].set_title('Equalized Histogram')
    else:
        colors = ('r', 'g', 'b')
        for ci, c in enumerate(colors):
            hist_o = histogram_from_channel(orig[..., ci])
            hist_e = histogram_from_channel(eq[..., ci])
            axes[1, 0].plot(hist_o, color=c)
            axes[1, 1].plot(hist_e, color=c)
        axes[1, 0].set_title('Original Histograms')
        axes[1, 1].set_title('Equalized Histograms')

    plt.tight_layout()
    plt.show()


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description='Pixelwise histogram equalization demo')
    parser.add_argument('--input', '-i', help='Input image file (optional)')
    parser.add_argument('--outdir', '-o', default='images', help='Output directory')
    parser.add_argument('--per-channel', dest='per_channel', action='store_true', help='Equalize per channel for color images')
    parser.add_argument('--no-per-channel', dest='per_channel', action='store_false', help='Use luminance mapping for color images')
    parser.set_defaults(per_channel=True)
    args = parser.parse_args()

    ensure_dir(args.outdir)

    samples = []
    if args.input:
        samples.append(args.input)
    else:
        samples.append(os.path.join('images', 'multimodal_hist.png'))
        samples.append(os.path.join('images', 'multimodal_hist_color.png'))

    for path in samples:
        if not os.path.exists(path):
            print(f'Skipping missing sample: {path}')
            continue

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f'Could not read {path}')
            continue

        # Convert BGRA->BGR and BGR->RGB for plotting/applying mapping consistently
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        if img.ndim == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            eq_rgb = equalize_image_pixelwise(img_rgb, per_channel=args.per_channel)
            # Save results (convert back to BGR for cv2.imwrite)
            base = os.path.splitext(os.path.basename(path))[0]
            out_eq = os.path.join(args.outdir, base + '_pixel_eq_color.png')
            cv2.imwrite(out_eq, cv2.cvtColor(eq_rgb, cv2.COLOR_RGB2BGR))
            print(f'Wrote {out_eq}')
            plot_side_by_side(img_rgb, eq_rgb, title_prefix=base + ' - ')
        else:
            # grayscale
            if img.ndim == 3:
                # single-channel images sometimes read as (H,W,1)
                img = img[..., 0]
            eq = equalize_image_pixelwise(img)
            base = os.path.splitext(os.path.basename(path))[0]
            out_eq = os.path.join(args.outdir, base + '_pixel_eq.png')
            cv2.imwrite(out_eq, eq)
            print(f'Wrote {out_eq}')
            plot_side_by_side(img, eq, title_prefix=base + ' - ')


if __name__ == '__main__':
    main()
