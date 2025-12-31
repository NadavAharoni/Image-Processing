import cv2

filename = "image.jpg"
img = cv2.imread(filename, cv2.IMREAD_COLOR)
if img is None:
    print(f"failed to read image from {filename}")
    exit(-2)
        
print((type(img)))
print(f"img.ndim={img.ndim}, img.shape={img.shape}, img.dtype={img.dtype}")
print(f"img[0,0]={img[0,0]}")

cv2.imshow(f"{filename}", img)
cv2.waitKey(0)
cv2.destroyAllWindows()