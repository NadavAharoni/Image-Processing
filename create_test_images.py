import numpy as np
import cv2
import matplotlib.pyplot as plt

def create_gradient_image(height, width):
    img = np.zeros((height, width), dtype=np.uint8)

    max_sum = (height - 1) + (width - 1)

    for r in range(height):
        for c in range(width):
            value = (r + c) * 255 / max_sum
            img[r, c] = int(value)

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

    fig, axes = plt.subplots(2, 1)
    axes[0].imshow(gradient_image, cmap='gray', vmin=0, vmax=255)
    axes[0].axis('off')

    axes[1].imshow(circle_image, cmap='gray', vmin=0, vmax=255)
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
