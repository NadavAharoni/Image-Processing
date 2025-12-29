import sys
import cv2

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} image-filename")
        exit(-1)

    file_name = sys.argv[1]
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)
    if img is None:
        print(f"failed to read image from {file_name}")
        exit(-2)
            
    cv2.imshow(f"{file_name}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print((type(img)))
    print(f"img.ndim={img.ndim}, img.shape={img.shape}")
    # print(img)

if __name__ == "__main__":
    main()
