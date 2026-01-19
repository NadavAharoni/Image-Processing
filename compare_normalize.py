import cv2
import numpy as np
import matplotlib.pyplot as plt
from create_test_images import create_rects

def normalize_per_channel(img):
    """Normalize each channel separately: min->0, max->255"""
    result = img.copy().astype(np.float32)
    for ch in range(result.shape[2]):
        ch_min = result[..., ch].min()
        ch_max = result[..., ch].max()
        if ch_max > ch_min:
            result[..., ch] = (result[..., ch] - ch_min) / (ch_max - ch_min) * 255
    return result.astype(np.uint8)

def stretch_around_mean_per_channel(img, factor=5):
    """Multiply distance from mean by factor for each channel separately"""
    result = img.copy().astype(np.float32)
    for ch in range(result.shape[2]):
        ch_data = result[..., ch]
        mean_val = ch_data.mean()
        ch_data = mean_val + (ch_data - mean_val) * factor
        ch_data = np.clip(ch_data, 0, 255)
        result[..., ch] = ch_data
    return result.astype(np.uint8)

def equalize_per_channel(img):
    """Histogram equalization per channel"""
    result = img.copy()
    for ch in range(result.shape[2]):
        result[..., ch] = cv2.equalizeHist(result[..., ch])
    return result

def normalize_hsv_brightness(img):
    """Convert to HSV, normalize brightness (V), convert back to RGB"""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v = hsv[..., 2].astype(np.float32)
    v_min = v.min()
    v_max = v.max()
    if v_max > v_min:
        v = (v - v_min) / (v_max - v_min) * 255
    hsv[..., 2] = np.clip(v, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return result

def stretch_hsv_brightness(img, factor=5):
    """Convert to HSV, stretch V channel around mean, convert back"""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v = hsv[..., 2].astype(np.float32)
    mean_v = v.mean()
    v = mean_v + (v - mean_v) * factor
    v = np.clip(v, 0, 255)
    hsv[..., 2] = v.astype(np.uint8)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return result

def equalize_hsv_brightness(img):
    """Convert to HSV, equalize V channel, convert back"""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[..., 2] = cv2.equalizeHist(hsv[..., 2])
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return result

def normalize_ycrcb_brightness(img):
    """Convert to YCrCb, normalize Y channel, convert back"""
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y = ycrcb[..., 0].astype(np.float32)
    y_min = y.min()
    y_max = y.max()
    if y_max > y_min:
        y = (y - y_min) / (y_max - y_min) * 255
    ycrcb[..., 0] = np.clip(y, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    return result

def stretch_ycrcb_brightness(img, factor=5):
    """Convert to YCrCb, stretch Y channel around mean, convert back"""
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y = ycrcb[..., 0].astype(np.float32)
    mean_y = y.mean()
    y = mean_y + (y - mean_y) * factor
    y = np.clip(y, 0, 255)
    ycrcb[..., 0] = y.astype(np.uint8)
    result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    return result

def equalize_ycrcb_brightness(img):
    """Convert to YCrCb, equalize Y channel, convert back"""
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycrcb[..., 0] = cv2.equalizeHist(ycrcb[..., 0])
    result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    return result

def main():
    # Create test image
    height, width = 300, 400
    bg_color = (120, 20, 20)
    
    # Calculate 1/8 of image area (approximately)
    rect_height = height // 4
    rect_width = width // 4
    
    # Two non-overlapping rectangles, each covering about 1/8 of image
    rect_list = [
        {'rect': ((50, 50), (50 + rect_height, 50 + rect_width)), 'color': (140, 21, 21)},
        {'rect': ((150, 250), (150 + rect_height, 250 + rect_width)), 'color': (150, 22, 22)}
    ]
    
    img_bgr = create_rects(height, width, 'RGB', bg_color, rect_list)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if isinstance(img_bgr, np.ndarray) else img_bgr
    
    # Apply all transformations
    transformations = [
        ("Original", img_rgb),
        ("1. Normalize per channel\n(min→0, max→255)", normalize_per_channel(img_rgb)),
        ("2. Stretch ×5 around mean\n(per channel)", stretch_around_mean_per_channel(img_rgb, factor=5)),
        ("3. Histogram equalization\n(per channel)", equalize_per_channel(img_rgb)),
        ("4. HSV normalize brightness", normalize_hsv_brightness(img_rgb)),
        ("5. HSV stretch brightness ×5", stretch_hsv_brightness(img_rgb, factor=5)),
        ("6. HSV histogram equalize", equalize_hsv_brightness(img_rgb)),
        ("7. YCrCb normalize brightness", normalize_ycrcb_brightness(img_rgb)),
        ("8. YCrCb stretch brightness ×5", stretch_ycrcb_brightness(img_rgb, factor=5)),
        ("9. YCrCb histogram equalize", equalize_ycrcb_brightness(img_rgb)),
    ]
    
    # Display results in a grid
    n_cols = 5
    n_rows = 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = axes.flatten()
    
    for idx, (title, img_result) in enumerate(transformations):
        ax = axes[idx]
        # Ensure image is in uint8 format
        img_display = np.clip(img_result, 0, 255).astype(np.uint8)
        ax.imshow(img_display)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
