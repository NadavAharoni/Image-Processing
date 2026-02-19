import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def create_grid(n=4, spacing=1.0):
    x = np.linspace(0, (n - 1) * spacing, n)
    y = np.linspace(0, (n - 1) * spacing, n)
    X, Y = np.meshgrid(x, y)
    ones = np.ones_like(X)
    points = np.vstack([X.ravel(), Y.ravel(), ones.ravel()])
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


def scale(s):
    return np.array([
        [s, 0, 0],
        [0, s, 0],
        [0, 0, 1]
    ])


def main():
    n = 4
    grid = create_grid(n)

    # Compute center of grid
    center_x = (n - 1) / 2
    center_y = (n - 1) / 2

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    fig.suptitle("Affine Transformation with Interactive Sliders")

    ax.set_aspect('equal')
    ax.set_xlim(-3, 6)
    ax.set_ylim(-3, 6)
    ax.grid(True)

    original_plot, = ax.plot(grid[0], grid[1], 'o')
    transformed_plot, = ax.plot(grid[0], grid[1], 'x')

    # Slider axes
    ax_rot = plt.axes([0.15, 0.1, 0.7, 0.03])
    ax_scale = plt.axes([0.15, 0.05, 0.7, 0.03])

    rot_slider = Slider(ax_rot, 'Rotation (deg)', -180, 180, valinit=0)
    scale_slider = Slider(ax_scale, 'Scale', 0.2, 2.0, valinit=1.0)
    
    def update(val):
        theta = np.deg2rad(rot_slider.val)
        s = scale_slider.val

        T1 = translation(-center_x, -center_y)
        R = rotation(theta)
        S = scale(s)
        T2 = translation(center_x, center_y)

        A = T2 @ R @ S @ T1
        transformed = A @ grid

        transformed_plot.set_data(transformed[0], transformed[1])
        fig.canvas.draw_idle()

    rot_slider.on_changed(update)
    scale_slider.on_changed(update)

    plt.show()


if __name__ == "__main__":
    main()
