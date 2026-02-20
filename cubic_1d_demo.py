import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


# -------------------------------------------------
# Cubic convolution kernel (Catmull–Rom style)
# -------------------------------------------------

def cubic_kernel(t, a=-0.5):
    t = abs(t)
    if t <= 1:
        return (a + 2)*t**3 - (a + 3)*t**2 + 1
    elif t < 2:
        return a*t**3 - 5*a*t**2 + 8*a*t - 4*a
    else:
        return 0


def cubic_interpolate(x, xs, ys):
    x0 = int(np.floor(x))
    result = 0.0

    for k in range(-1, 3):
        idx = np.clip(x0 + k, 0, len(ys) - 1)
        result += ys[idx] * cubic_kernel(x - (x0 + k))

    return result


# -------------------------------------------------
# Main demo
# -------------------------------------------------

def main():

    xs = np.array([0, 1, 2, 3], dtype=float)
    ys = np.array([1, 3, 2, 4], dtype=float)

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.30, bottom=0.15)

    ax.set_xlim(-0.2, 3.2)

    margin = 1.5
    ax.set_ylim(-5 - margin, 5 + margin)
    
    ax.grid(True)
    ax.set_title("1D Cubic Interpolation (4 Points)")

    # Dense curve
    x_dense = np.linspace(0, 3, 400)

    cubic_line, = ax.plot([], [], label="Cubic")
    linear_line, = ax.plot([], [], '--', label="Linear")
    points_plot, = ax.plot(xs, ys, 'ro')

    ax.legend()

    # -------------------------------------------------
    # Vertical sliders (one per point)
    # -------------------------------------------------

    slider_axes = []
    sliders = []

    for i in range(4):
        ax_slider = plt.axes([0.05 + i*0.05, 0.25, 0.03, 0.5])
        slider = Slider(
            ax_slider,
            '',
            -5, 5,
            valinit=ys[i],
            orientation='vertical'
        )
        slider_axes.append(ax_slider)
        sliders.append(slider)

    # -------------------------------------------------
    # Update function
    # -------------------------------------------------

    def update(val):
        for i in range(4):
            ys[i] = sliders[i].val

        y_cubic = np.array([cubic_interpolate(x, xs, ys) for x in x_dense])
        y_linear = np.interp(x_dense, xs, ys)

        cubic_line.set_data(x_dense, y_cubic)
        linear_line.set_data(x_dense, y_linear)
        points_plot.set_data(xs, ys)

        fig.canvas.draw_idle()

    for s in sliders:
        s.on_changed(update)

    update(None)
    plt.show()


if __name__ == "__main__":
    main()
