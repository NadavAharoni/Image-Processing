import os
import cv2
import argparse
import sys

def main():
    # 1. Setup argparse to handle command-line arguments
    parser = argparse.ArgumentParser(description="Convert an image to grayscale.")
    parser.add_argument("input_file", help="Path to the input image file")
    args = parser.parse_args()

    # 2. Load the image
    # Note: Using cv2.IMREAD_GRAYSCALE (or 0) loads it as grayscale directly
    image = cv2.imread(args.input_file)

    if image is None:
        print(f"Error: Could not open or find the image '{args.input_file}'")
        sys.exit(1)

    # 3. Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 4. Save the image to disk
    file, ext = os.path.splitext(args.input_file)
    print(f"file={file}, ext={ext}")
    output_filename = f'{file}_grayscale_cv2{ext}'
    success = cv2.imwrite(output_filename, gray_image)

    if success:
        print(f"Successfully saved: {output_filename}")
    else:
        print("Error: Could not save the image.")

if __name__ == "__main__":
    main()
