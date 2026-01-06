import matplotlib.pyplot as plt
import numpy as np
import cv2

import create_test_images
import brightten

def main():
    height = 300
    width = 400

    # create a "plot" with two rows and 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(10, 4))

    gradient_image = create_test_images.create_gradient_image(height, width)
    axes[0,0].imshow(gradient_image, cmap='gray')
    axes[0,0].set_title('Original')
    axes[0,0].axis('off')
    
    brightened_image_np = brightten.brighten(gradient_image, 50, "np")
    axes[1,0].imshow(brightened_image_np, cmap='gray', vmin=0, vmax=255)
    axes[1,0].set_title('Brightened (np)')
    axes[1,0].axis('off')

    brightened_image_cv2 = brightten.brighten(gradient_image, 50, "cv2")
    axes[0,1].imshow(brightened_image_cv2, cmap='gray', vmin=0, vmax=255)
    axes[0,1].set_title('Brightened (cv2)')
    axes[0,1].axis('off')

    axes[1,1].axis('off')

    # low_contrast_image = create_test_images.create_circle_image(height, width, 32, 35)
    low_contrast_image = create_test_images.create_hidden_text_image(height, width)
    axes[0,2].imshow(low_contrast_image, cmap='gray', vmin=0, vmax=255)
    axes[0,2].set_title('low contrast')
    axes[0,2].axis('off')

    min_val = np.min(low_contrast_image)
    max_val = np.max(low_contrast_image)
    print(min_val, max_val)
    image_normalized = (low_contrast_image - min_val) * ( 255 / (max_val - min_val) )
    axes[1,2].imshow(image_normalized, cmap='gray', vmin=0, vmax=255)
    axes[1,2].set_title('normalized')
    axes[1,2].axis('off')

    axes[1,3].axis('off')
    axes[0,3].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()