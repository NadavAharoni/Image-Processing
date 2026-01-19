import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def create_rects():
    """
    Parameters:
    height, width, type, bg, rect_list

    Creates an image with size (height, width)
    and background color bg.
    
    The "type" argument determines if the image
    is greyscale, RGB, etc.
    
    Then draws on the image rectangles according to
    the spec in "rect_list".
    "rect_list contains a list of dicts, each with the structure:
    { 'rect':((r0,c0),(r1,c1)), 'color':c }
    Where the value of 'rect' are the 
    top-right and bottom-left corners of the rectanle,
    and c is the color of the recatanlge.
    """
    pass

def create_gradient_image(height, width):
    img = np.zeros((height, width), dtype=np.uint8)

    max_sum = (height - 1) + (width - 1)

    for r in range(height):
        for c in range(width):
            value = (r + c) * 255 / max_sum
            img[r, c] = int(value)

    return img


def draw_gradient_rect(img, rect, col1, col2):
    """Draw a filled rectangle with a gradient from color col1 (top-left)
    to color col2 (bottom-right) onto `img` in-place.

    - img: numpy array, either 2D (grayscale) or 3D (H,W,3) with dtype uint8
    - rect: ((r0, c0), (r1, c1)) top-left and bottom-right
    - col1, col2: grayscale int (0..255) or RGB tuples of length 3
    """
    (r0, c0), (r1, c1) = rect

    # Clip coordinates to image bounds
    height, width = img.shape[:2]
    r0 = max(0, int(r0))
    c0 = max(0, int(c0))
    r1 = min(height, int(r1))
    c1 = min(width, int(c1))

    if r0 >= r1 or c0 >= c1:
        return  # nothing to draw

    # Determine if color or grayscale
    is_color = img.ndim == 3 and img.shape[2] == 3

    # Normalize colors to arrays
    if is_color:
        c1_arr = np.array(col1, dtype=np.float32)
        c2_arr = np.array(col2, dtype=np.float32)
    else:
        c1_val = float(col1)
        c2_val = float(col2)

    h = r1 - r0
    w = c1 - c0

    # Avoid division by zero
    denom_r = h - 1 if h > 1 else 1
    denom_c = w - 1 if w > 1 else 1

    for rr in range(r0, r1):
        fr = (rr - r0) / denom_r
        for cc in range(c0, c1):
            fc = (cc - c0) / denom_c
            t = (fr + fc) / 2.0
            if is_color:
                val = (1.0 - t) * c1_arr + t * c2_arr
                img[rr, cc] = np.clip(val, 0, 255).astype(np.uint8)
            else:
                val = (1.0 - t) * c1_val + t * c2_val
                img[rr, cc] = int(np.clip(val, 0, 255))


def create_multimodal_hist_image(specs, height, width, color=False):
    """Create an image composed of gradient rectangles.

    - specs: list of dicts, each with keys: 'rect':((r0,c0),(r1,c1)), 'colors':(c1,c2)
    - color: if True create RGB image, else grayscale
    """
    if color:
        img = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        img = np.zeros((height, width), dtype=np.uint8)

    for spec in specs:
        rect = spec['rect']
        c1, c2 = spec['colors']
        draw_gradient_rect(img, rect, c1, c2)

    return img


def create_circle_image(height, width, bg=128, fg=130):
    """
    Creates a grayscale image with:
    - background intensity = bg
    - a circle with intensity = fg
    """

    # Create image filled with background value
    img = np.full((height, width), fill_value=bg, dtype=np.uint8)

    # Circle parameters
    center = (width // 2, height // 2)
    radius = min(height, width) // 4

    # Draw circle
    cv2.circle(
        img,
        center=center,
        radius=radius,
        color=fg,     # grayscale value
        thickness=-1   # filled circle
    )

    return img


def create_hidden_text_image(height, width, text="password"):
    """
    Creates a grayscale image with:
    - background intensity = 127
    - text intensity = 128
    """
    img = np.full((height, width), 127, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    color = 128  # text intensity

    # Get text size to center it
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    x = (width - text_width) // 2
    y = (height + text_height) // 2

    cv2.putText(
        img,
        text,
        org=(x, y),
        fontFace=font,
        fontScale=font_scale,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA
    )

    return img


if __name__ == "__main__":
    height = 400
    width = 500

    gradient_image = create_gradient_image(height, width)
    circle_image = create_circle_image(height, width)

    print(circle_image.shape)
    cv2.imwrite(f'images\\low_contrast_circle.png', circle_image)

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(gradient_image, cmap='gray', vmin=0, vmax=255)
    axes[0].axis('off')

    axes[1].imshow(circle_image, cmap='gray', vmin=0, vmax=255)
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    # --- create example image for histogram demonstration ---
    specs = [
        # A: whole image low near 50..60
        {'rect': ((0, 0), (height, width)), 'colors': (50, 60)},
        # B: a non-overlapping area with slightly higher values
        {'rect': ((30, 30), (220, 200)), 'colors': (70, 80)},
        # C: a bright region on the right
        {'rect': ((100, 300), (360, 480)), 'colors': (180, 185)},
        # D: a sub-area of C, even brighter
        {'rect': ((150, 340), (260, 430)), 'colors': (195, 200)},
    ]

    hist_img = create_multimodal_hist_image(specs, height, width, color=False)
    out_path = os.path.join('images', 'multimodal_hist.png')
    cv2.imwrite(out_path, hist_img)
    print(f'Wrote {out_path}')

    # --- Step 2: create color example and demonstrate equalization ---
    color_specs = [
        {'rect': ((0, 0), (height, width)), 'colors': ((50, 20, 30), (55, 40, 35))},
        {'rect': ((30, 30), (220, 200)), 'colors': ((70, 10, 20), (80, 40, 30))},
        {'rect': ((100, 300), (360, 480)), 'colors': ((180, 120, 130), (185, 140, 135))},
        {'rect': ((150, 340), (260, 430)), 'colors': ((195, 150, 160), (200, 155, 165))},
    ]

    color_img = create_multimodal_hist_image(color_specs, height, width, color=True)

    def equalize_per_channel_rgb(img_rgb):
        out = img_rgb.copy()
        for ch in range(3):
            out[..., ch] = cv2.equalizeHist(out[..., ch])
        return out

    def equalize_value_channel_hsv(img_rgb):
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        hsv[..., 2] = cv2.equalizeHist(hsv[..., 2])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def equalize_luminance_ycrcb(img_rgb):
        ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
        ycrcb[..., 0] = cv2.equalizeHist(ycrcb[..., 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

    # Save original color image
    out_color = os.path.join('images', 'multimodal_hist_color.png')
    cv2.imwrite(out_color, cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR))

    # Equalize per-channel (may change colors undesirably)
    eq_rgb = equalize_per_channel_rgb(color_img)
    out_eq_rgb = os.path.join('images', 'multimodal_hist_color_eq_rgb.png')
    cv2.imwrite(out_eq_rgb, cv2.cvtColor(eq_rgb, cv2.COLOR_RGB2BGR))

    # Equalize V (brightness) in HSV
    eq_hsv = equalize_value_channel_hsv(color_img)
    out_eq_hsv = os.path.join('images', 'multimodal_hist_color_eq_hsv.png')
    cv2.imwrite(out_eq_hsv, cv2.cvtColor(eq_hsv, cv2.COLOR_RGB2BGR))

    # Equalize luminance in YCrCb
    eq_y = equalize_luminance_ycrcb(color_img)
    out_eq_y = os.path.join('images', 'multimodal_hist_color_eq_ycrcb.png')
    cv2.imwrite(out_eq_y, cv2.cvtColor(eq_y, cv2.COLOR_RGB2BGR))

    print(f'Wrote {out_color}, {out_eq_rgb}, {out_eq_hsv}, {out_eq_y}')
