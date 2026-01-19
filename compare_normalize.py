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
    bg_color = (120, 30, 25)
    
    # Calculate 1/8 of image area (approximately)
    rect_height = height // 4
    rect_width = width // 4
    
    # Two non-overlapping rectangles, each covering about 1/8 of image
    rect_list = [
        {'rect': ((50, 50), (50 + rect_height, 50 + rect_width)), 'color': (135, 31, 22)},
        {'rect': ((150, 250), (150 + rect_height, 250 + rect_width)), 'color': (150, 20, 30)},
        {'rect': ((110, 160), (110 + 30, 160 + 40)), 'color': (100, 10, 29)}
    ]
    
    img = create_rects(height, width, 'RGB', bg_color, rect_list)
    
    # Prepare transformations organized by method
    per_channel_transforms = [
        ("Normalize\n(min→0, max→255)", normalize_per_channel(img)),
        ("Stretch ×5\naround mean", stretch_around_mean_per_channel(img, factor=5)),
        ("Histogram\nequalization", equalize_per_channel(img)),
    ]
    
    hsv_transforms = [
        ("Normalize\nbrightness", normalize_hsv_brightness(img)),
        ("Stretch brightness\n×5", stretch_hsv_brightness(img, factor=5)),
        ("Histogram\nequalize", equalize_hsv_brightness(img)),
    ]
    
    ycrcb_transforms = [
        ("Normalize\nbrightness", normalize_ycrcb_brightness(img)),
        ("Stretch brightness\n×5", stretch_ycrcb_brightness(img, factor=5)),
        ("Histogram\nequalize", equalize_ycrcb_brightness(img)),
    ]
    
    # Display results in a grid
    fig, axes = plt.subplots(4, 3, figsize=(8, 9))
    
    # Row 0: Original image (centered)
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    axes[0, 1].axis('off')
    axes[0, 2].axis('off')
    
    # Row 1: Per-channel transformations
    for col, (title, img_result) in enumerate(per_channel_transforms):
        ax = axes[1, col]
        img_display = np.clip(img_result, 0, 255).astype(np.uint8)
        ax.imshow(img_display)
        ax.set_title(f"Per-channel: {title}", fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # Row 2: HSV transformations
    for col, (title, img_result) in enumerate(hsv_transforms):
        ax = axes[2, col]
        img_display = np.clip(img_result, 0, 255).astype(np.uint8)
        ax.imshow(img_display)
        ax.set_title(f"HSV: {title}", fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # Row 3: YCrCb transformations
    for col, (title, img_result) in enumerate(ycrcb_transforms):
        ax = axes[3, col]
        img_display = np.clip(img_result, 0, 255).astype(np.uint8)
        ax.imshow(img_display)
        ax.set_title(f"YCrCb: {title}", fontsize=10, fontweight='bold')
        ax.axis('off')
    
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
