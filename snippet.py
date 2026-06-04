base_train_transforms = [
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ]

if args.augment:
    base_train_transforms = [
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    ] + base_train_transforms

train_transform = transforms.Compose(base_train_transforms)


# Block 2: conv → ReLU → maxpool
# Input:  conv1_filters × 14 × 14
# Output: conv2_filters × 7 × 7
self.block2 = nn.Sequential(
    nn.Conv2d(in_channels=conv1_filters,
              out_channels=conv2_filters,
              kernel_size=3, padding=1),
    nn.ReLU(),
    
    nn.MaxPool2d(kernel_size=2, stride=2)
    # → 7×7
)



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

# ===========
import numpy as np
c = np.array((200,100,100)) / 255.0   
v = np.max(c)
delta = v - np.min(c)
s = delta / v

print(f"c={c}")
print(f"v={v}")
print(f"delta={delta}")
print(f"s={s}")
