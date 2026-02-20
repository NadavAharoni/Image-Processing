import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def create_grid(n=4, spacing=1.0):
    x = np.linspace(0, (n - 1) * spacing, n)
    y = np.linspace(0, (n - 1) * spacing, n)
    X, Y = np.meshgrid(x, y)
    ones = np.ones_like(X)
    return np.vstack([X.ravel(), Y.ravel(), ones.ravel()])


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


def main():
    n = 4
    grid = create_grid(n)

    center_x = (n - 1) / 2
    center_y = (n - 1) / 2

    fig, ax = plt.subplots()
    fig.suptitle("Affine Transformation with Order Switch")

    # Leave space for sliders
    plt.subplots_adjust(bottom=0.32)

    # Square axes (independent of window size)
    ax.set_box_aspect(1)

    # Tight limits so grid fills the area nicely
    margin = 0.5
    ax.set_xlim(-margin, (n - 1) + margin)
    ax.set_ylim(-margin, (n - 1) + margin)
    ax.set_xticks(np.arange(0, n, 1))
    ax.set_yticks(np.arange(0, n, 1))
    ax.grid(True)

    # Original and transformed grids
    ax.plot(grid[0], grid[1], 'o')
    transformed_plot, = ax.plot(grid[0], grid[1], 'x')

    # ---- Sliders ----
    ax_rot   = plt.axes([0.15, 0.22, 0.7, 0.03])
    ax_sx    = plt.axes([0.15, 0.17, 0.7, 0.03])
    ax_sy    = plt.axes([0.15, 0.12, 0.7, 0.03])
    ax_order = plt.axes([0.15, 0.05, 0.12, 0.04])

    rot_slider = Slider(ax_rot, 'Rotation (deg)', -180, 180, valinit=0)
    sx_slider = Slider(ax_sx, 'Scale X', 0.2, 2.0, valinit=1.0)
    sy_slider = Slider(ax_sy, 'Scale Y', 0.2, 2.0, valinit=1.0)

    order_slider = Slider(
        ax_order,
        '',
        0, 1,
        valinit=0,
        valstep=1
    )

    ax_order.set_title("R·S   |   S·R", fontsize=9)

    # ---- Update function ----
    def update(val):
        theta = np.deg2rad(rot_slider.val)
        sx = sx_slider.val
        sy = sy_slider.val
        order = int(order_slider.val)

        T1 = translation(-center_x, -center_y)
        R = rotation(theta)
        S = scale(sx, sy)
        T2 = translation(center_x, center_y)

        if order == 0:
            A = T2 @ R @ S @ T1
        else:
            A = T2 @ S @ R @ T1

        transformed = A @ grid
        transformed_plot.set_data(transformed[0], transformed[1])

        fig.canvas.draw_idle()

    rot_slider.on_changed(update)
    sx_slider.on_changed(update)
    sy_slider.on_changed(update)
    order_slider.on_changed(update)

    plt.show()


if __name__ == "__main__":
    main()
