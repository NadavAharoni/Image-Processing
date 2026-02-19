import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------
# 1. Create original 4x4 grid
# ---------------------------------
n = 4
x = np.linspace(0, 3, n)
y = np.linspace(0, 3, n)
X, Y = np.meshgrid(x, y)

points = np.vstack([X.ravel(), Y.ravel()])  # shape (2, 16)

# ---------------------------------
# 2. Define a transformation
#    (Example: rotation + scaling + shear)
# ---------------------------------
theta = np.deg2rad(30)
scale = 1.2

T = np.array([
    [scale * np.cos(theta), -np.sin(theta) + 0.3],
    [np.sin(theta), scale * np.cos(theta)]
])

# Apply transformation
transformed_points = T @ points

# ---------------------------------
# 3. Plot original and transformed
# ---------------------------------
plt.figure()
plt.scatter(points[0], points[1], marker='o', label='Original')
plt.scatter(transformed_points[0], transformed_points[1], marker='x', label='Transformed')

# Draw correspondence lines
for i in range(points.shape[1]):
    plt.plot(
        [points[0, i], transformed_points[0, i]],
        [points[1, i], transformed_points[1, i]]
    )

plt.axhline(0)
plt.axvline(0)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.legend()
plt.title("Forward Transformation of Grid Points")
plt.show()
