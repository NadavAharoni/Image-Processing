import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt


# -----------------------------
# Interpolation Functions
# -----------------------------

def nearest_neighbor(img, x, y):
    x_round = int(round(x))
    y_round = int(round(y))
    
    if x_round < 0 or y_round < 0 or \
       x_round >= img.shape[1] or \
       y_round >= img.shape[0]:
        return 0
    
    return img[y_round, x_round]


def bilinear(img, x, y):
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    
    x1 = x0 + 1
    y1 = y0 + 1
    
    if x0 < 0 or y0 < 0 or \
       x1 >= img.shape[1] or \
       y1 >= img.shape[0]:
        return 0
    
    dx = x - x0
    dy = y - y0
    
    I00 = img[y0, x0]
    I10 = img[y0, x1]
    I01 = img[y1, x0]
    I11 = img[y1, x1]
    
    return (I00 * (1 - dx) * (1 - dy) +
            I10 * dx * (1 - dy) +
            I01 * (1 - dx) * dy +
            I11 * dx * dy)


# -----------------------------
# Homogeneous Transform Builder
# -----------------------------

def build_transform(scale=1.0, angle_deg=0.0, tx=0.0, ty=0.0):
    theta = np.deg2rad(angle_deg)
    
    S = np.array([[scale, 0, 0],
                  [0, scale, 0],
                  [0, 0, 1]])
    
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta),  np.cos(theta), 0],
                  [0, 0, 1]])
    
    T = np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0, 1]])
    
    # Order: first scale, then rotate, then translate
    return T @ R @ S


# -----------------------------
# Backward Mapping
# -----------------------------

def apply_transform(img, M, interpolation='nearest'):
    h, w = img.shape
    output = np.zeros_like(img)
    
    M_inv = np.linalg.inv(M)
    
    cx = w // 2
    cy = h // 2
    
    for y in range(h):
        for x in range(w):
            
            # Move origin to center
            p = np.array([x - cx, y - cy, 1])
            
            # Backward mapping
            source = M_inv @ p
            
            xs = source[0] + cx
            ys = source[1] + cy
            
            if interpolation == 'nearest':
                output[y, x] = nearest_neighbor(img, xs, ys)
            elif interpolation == 'bilinear':
                output[y, x] = bilinear(img, xs, ys)
    
    return output


# -----------------------------
# Demo
# -----------------------------

def main(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    M1 = build_transform(scale=1.0, angle_deg=30, tx=0, ty=0)
    M2 = build_transform(scale=1.0, angle_deg=30, tx=50, ty=0)
    M3 = build_transform(scale=1.5, angle_deg=45, tx=30, ty=20)
    
    out1 = apply_transform(img, M1, 'nearest')
    out2 = apply_transform(img, M2, 'bilinear')
    out3 = apply_transform(img, M3, 'bilinear')
    
    plt.figure(figsize=(12,6))
    
    plt.subplot(1,4,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    
    plt.subplot(1,4,2)
    plt.imshow(out1, cmap='gray')
    plt.title("Rotate 30°")
    
    plt.subplot(1,4,3)
    plt.imshow(out2, cmap='gray')
    plt.title("Rotate + Translate")
    
    plt.subplot(1,4,4)
    plt.imshow(out3, cmap='gray')
    plt.title("Scale + Rotate + Translate")
    
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transformations_01.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    main(image_path)
