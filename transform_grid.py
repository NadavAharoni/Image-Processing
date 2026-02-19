import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def create_grid(n=4, spacing=1.0):
    x = np.linspace(0, (n - 1) * spacing, n)
    y = np.linspace(0, (n - 1) * spacing, n)
    X, Y = np.meshgrid(x, y)
    ones = np.ones_like(X)
    points = np.vstack([X.ravel(), Y.ravel(), ones.ravel()])  # homogeneous (3, N)
    return points


# -----------------------------
# Affine matrices (3x3)
# -----------------------------

def translation(tx, ty):
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])


def rotation(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])


def scale(sx, sy):
    return np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ])


# -----------------------------
# Main animation
# -----------------------------

def main():
    n = 4
    grid = create_grid(n)

    # Compute center of grid
    center_x = (n - 1) / 2
    center_y = (n - 1) / 2

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(-2, 5)
    ax.set_ylim(-2, 5)
    ax.grid(True)

    original_plot, = ax.plot(grid[0], grid[1], 'o')
    transformed_plot, = ax.plot([], [], 'x')

    def update(frame):
        theta = np.deg2rad(frame)

        # Build matrices individually
        T1 = translation(-center_x, -center_y)   # move center to origin
        R = rotation(theta)                      # rotate
        S = scale(1.0, 1.0)                      # optional scaling (identity here)
        T2 = translation(center_x, center_y)     # move back

        # Combine: T2 * R * S * T1
        A = T2 @ R @ S @ T1

        transformed = A @ grid

        transformed_plot.set_data(transformed[0], transformed[1])
        return transformed_plot,

    anim = FuncAnimation(fig, update, frames=np.linspace(0, 360, 120), interval=50)

    plt.title("Affine Rotation Around Grid Center")
    plt.show()


if __name__ == "__main__":
    main()
