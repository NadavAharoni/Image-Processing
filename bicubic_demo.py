import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# -------------------------------------------------
# Sample 4x4 "image" (16 pixel heights)
# -------------------------------------------------

Z_pixels = np.array([
    [10, 20, 30, 20],
    [15, 25, 35, 30],
    [20, 30, 40, 35],
    [15, 25, 30, 20]
], dtype=float)

H, W = Z_pixels.shape


# -------------------------------------------------
# Bilinear interpolation (cell-based)
# -------------------------------------------------

def bilinear(x, y):
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, W - 1)
    y1 = min(y0 + 1, H - 1)

    dx = x - x0
    dy = y - y0

    f00 = Z_pixels[y0, x0]
    f10 = Z_pixels[y0, x1]
    f01 = Z_pixels[y1, x0]
    f11 = Z_pixels[y1, x1]

    return (
        f00 * (1 - dx) * (1 - dy) +
        f10 * dx * (1 - dy) +
        f01 * (1 - dx) * dy +
        f11 * dx * dy
    )


# -------------------------------------------------
# Bicubic interpolation (cubic convolution)
# -------------------------------------------------

def cubic_kernel(t, a=-0.5):
    t = abs(t)
    if t <= 1:
        return (a + 2)*t**3 - (a + 3)*t**2 + 1
    elif t < 2:
        return a*t**3 - 5*a*t**2 + 8*a*t - 4*a
    else:
        return 0


def bicubic(x, y):
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))

    result = 0.0
    for m in range(-1, 3):
        for n in range(-1, 3):
            xm = np.clip(x0 + m, 0, W - 1)
            yn = np.clip(y0 + n, 0, H - 1)
            weight = cubic_kernel(x - (x0 + m)) * cubic_kernel(y - (y0 + n))
            result += Z_pixels[yn, xm] * weight
    return result


# -------------------------------------------------
# Create dense evaluation grid (0.1 spacing)
# -------------------------------------------------

step = 0.1
xs = np.arange(0, W - 1, step)
ys = np.arange(0, H - 1, step)

X_dense, Y_dense = np.meshgrid(xs, ys)

Z_bilinear = np.zeros_like(X_dense)
Z_bicubic = np.zeros_like(X_dense)

for i in range(X_dense.shape[0]):
    for j in range(X_dense.shape[1]):
        x = X_dense[i, j]
        y = Y_dense[i, j]
        Z_bilinear[i, j] = bilinear(x, y)
        Z_bicubic[i, j] = bicubic(x, y)


# -------------------------------------------------
# Plotting
# -------------------------------------------------

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# Bilinear surface
ax1.plot_surface(X_dense, Y_dense, Z_bilinear, cmap='viridis', alpha=0.8)
ax1.scatter(*np.meshgrid(np.arange(W), np.arange(H)),
            Z_pixels, color='red', s=40)
ax1.set_title("Bilinear Interpolation")

# Bicubic surface
ax2.plot_surface(X_dense, Y_dense, Z_bicubic, cmap='viridis', alpha=0.8)
ax2.scatter(*np.meshgrid(np.arange(W), np.arange(H)),
            Z_pixels, color='red', s=40)
ax2.set_title("Bicubic Interpolation")

for ax in [ax1, ax2]:
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("value")

plt.tight_layout()
plt.show()
