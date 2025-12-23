import sys
import PIL.Image # Image is a module, not a class
# from PIL import Image

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} image-filename")
        exit(-1)

    file_name = sys.argv[1]
    img = PIL.Image.open(file_name)
    img.show("original image")

    size = img.size
    print(size)
    
    # get pixel values through the getpixel function
    pixel = img.getpixel((0,0))
    print(pixel)

    middle = size[0]//2, size[1]//2
    print( img.getpixel(middle) )

    # get pixel values through a PixelAccess object
    # img.load() returns a PixelAccess object
    px = img.load()
    print(middle)
    if px is not None:
        print( px[0,0] )
        print( px[middle])

    # display the red channel
    r, g, b = img.split()
    blank = PIL.Image.new('L', size, 0)
    img_red = PIL.Image.merge("RGB", (r, blank, blank))
    print(img_red.getpixel((0,0)))
    img_red.show("red")

if __name__ == "__main__":
    main()
