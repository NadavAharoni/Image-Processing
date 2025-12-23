import sys
import PIL.Image # Image is a module, not a class
# from PIL import Image

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} image-filename")
        exit(-1)

    file_name = sys.argv[1]
    img = PIL.Image.open(file_name)
    # img.show()

    size = img.size
    print(size)
    pixel = img.getpixel((0,0))
    print(pixel)

    px = img.load()

    if px is not None:
        print(px[size[0]//2,size[1]//2])


if __name__ == "__main__":
    main()
