import sys
import os
from PIL import Image, ImageOps

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} image-filename")
        exit(-1)

    # 1. Load the original RGB image
    filename = sys.argv[1]
    img = Image.open(filename)
    print(f"img.mode={img.mode}")

    # 2. Convert to YCbCr color space (Y = Luminance/Brightness)
    ycbcr_img = img.convert('YCbCr')

    # 3. Split the channels
    y, cb, cr = ycbcr_img.split()

    # 4. Equalize ONLY the Y (Luminance) channel
    # This enhances contrast without distorting the actual colors (Cb/Cr)
    y_equalized = ImageOps.equalize(y)

    # 5. Merge the equalized Y channel back with the original Cb and Cr channels
    equalized_ycbcr = Image.merge('YCbCr', (y_equalized, cb, cr))

    # 6. Convert back to RGB for viewing/saving
    result_img = equalized_ycbcr.convert('RGB')

    # 7. Save or display the result
    file, ext = os.path.splitext(filename)
    result_img.save(f'{file}_equalized_pil.{ext}')
    # result_img.show()

if __name__ == "__main__":
    main()
