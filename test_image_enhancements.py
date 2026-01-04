import matplotlib.pyplot as plt

import create_test_images
import brightten

def main():
    height = 300
    width = 400
    fig, axes = plt.subplots(2, 2)

    gradient_image = create_test_images.create_gradient_image(height, width)
    axes[0,0].imshow(gradient_image, cmap='gray')
    axes[0,0].set_title('Original')
    axes[0,0].axis('off')
    
    brightened_image_np = brightten.brighten(gradient_image, 50, "np")
    axes[1,0].imshow(brightened_image_np, cmap='gray')
    axes[1,0].set_title('Brightened (np)')
    axes[1,0].axis('off')

    brightened_image_cv2 = brightten.brighten(gradient_image, 50, "cv2")
    axes[0,1].imshow(brightened_image_cv2, cmap='gray')
    axes[0,1].set_title('Brightened (cv2)')
    axes[0,1].axis('off')

    axes[1,1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()