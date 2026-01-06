import sys
import os
import cv2

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} image-filename")
        exit(-1)

    # 1. Load the image in BGR format (OpenCV default)
    file_name = sys.argv[1]
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)
    if img is None:
        print(f"failed to read image from {file_name}")
        exit(-2)

    # 2. Convert from BGR to YCrCb color space
    # Y = Luminance, Cr = Red-difference Chroma, Cb = Blue-difference Chroma
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # 3. Split the YCrCb image into its three channels
    # ycrcb_img[:, :, 0] is the Y channel
    channels = list(cv2.split(ycrcb_img))

    # 4. Equalize the histogram of the Y (Luminance) channel only
    channels[0] = cv2.equalizeHist(channels[0])

    # 5. Merge the equalized Y channel back with the original Cr and Cb channels
    ycrcb_equalized = cv2.merge(channels)

    # 6. Convert the YCrCb image back to BGR format
    result_img = cv2.cvtColor(ycrcb_equalized, cv2.COLOR_YCrCb2BGR)

    # 7. Save or display the result
    file, ext = os.path.splitext(file_name)
    cv2.imwrite(f'{file}_equalized_cv2{ext}', result_img)
    
    # cv2.imshow('Equalized Image', result_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()    

if __name__ == "__main__":
    main()
