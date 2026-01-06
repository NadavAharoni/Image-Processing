import sys
import os
from PIL import Image, ImageEnhance, ImageOps

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} image-filename")
        exit(-1)

    # Open an image
    filename = sys.argv[1]
    img = Image.open(filename)

    # Apply the rotation based on EXIF data
    img = ImageOps.exif_transpose(img)

    # Create a brightness enhancer object
    enhancer = ImageEnhance.Brightness(img)

    # Adjust the brightness with an enhancement factor
    # Factor = 1.0 gives the original image
    # Factor < 1.0 makes the image darker (0.0 makes it black)
    # Factor > 1.0 makes the image brighter (e.g., 1.5 increases brightness by 50%)
    brightness_factor = 1.1
    result_img = enhancer.enhance(brightness_factor)

    # Save the new image
    file, ext = os.path.splitext(filename)
    brightness_str = str(brightness_factor).replace('.','_')
    result_img.save(f'{file}_brightness_{brightness_str}{ext}')

    # Optional: Display the new image
    # bright_img.show()


if __name__ == "__main__":
    main()
