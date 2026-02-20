import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def bilinear(x, y, f00, f10, f01, f11):
    return (
        f00 * (1 - x) * (1 - y) +
        f10 * x * (1 - y) +
        f01 * (1 - x) * y +
        f11 * x * y
    )


def main():
    f00, f10 = 10, 20
    f01, f11 = 30, 40

    fig, ax = plt.subplots()
    fig.suptitle("Bilinear Interpolation", y=0.96)

    plt.subplots_adjust(bottom=0.35, top=0.90)

    ax.set_box_aspect(1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.grid(True)

    # Draw square
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0])

    # Corner labels slightly outside square
    offset = 0.12  # increased spacing

    ax.text(0, -offset, f"f00={f00}",
            ha="right", va="top", clip_on=False)

    ax.text(1, -offset, f"f10={f10}",
            ha="left", va="top", clip_on=False)

    ax.text(0, 1 + offset, f"f01={f01}",
            ha="right", va="bottom", clip_on=False)

    ax.text(1, 1 + offset, f"f11={f11}",
            ha="left", va="bottom", clip_on=False)

    point_plot, = ax.plot([0.5], [0.5], 'ro')
    h_line, = ax.plot([], [], '--')
    v_line, = ax.plot([], [], '--')

    # Place interpolated value BELOW the axes (cleanest solution)
    value_text = fig.text(0.5, 0.30, "", ha="center")

    # Sliders
    ax_x = plt.axes([0.2, 0.18, 0.6, 0.03])
    ax_y = plt.axes([0.2, 0.13, 0.6, 0.03])

    slider_x = Slider(ax_x, "x", 0, 1, valinit=0.5)
    slider_y = Slider(ax_y, "y", 0, 1, valinit=0.5)

    def update(val):
        x = slider_x.val
        y = slider_y.val

        fxy = bilinear(x, y, f00, f10, f01, f11)

        point_plot.set_data([x], [y])
        h_line.set_data([0, 1], [y, y])
        v_line.set_data([x, x], [0, 1])

        value_text.set_text(f"Interpolated value: {fxy:.2f}")

        fig.canvas.draw_idle()

    slider_x.on_changed(update)
    slider_y.on_changed(update)

    update(None)
    plt.show()


if __name__ == "__main__":
    main()
