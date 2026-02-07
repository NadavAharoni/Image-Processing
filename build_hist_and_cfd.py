import numpy as np
import matplotlib.pyplot as plt

def build_histogram(ranges, value=10):
    """
    ranges: list of (start, end) inclusive
    value: frequency for each bin in range
    """
    hist = np.zeros(256, dtype=int)
    for start, end in ranges:
        hist[start:end+1] = value
    return hist

def compute_cdf(hist):
    cdf = np.cumsum(hist)
    if cdf.max() > 0:
        cdf = cdf / cdf.max()
    return cdf

def plot_hist_and_cdf(hist, title):
    cdf = compute_cdf(hist)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5))

    ax1.bar(range(256), hist, width=1.0)
    ax1.set_title(f"{title} – Histogram")
    ax1.set_xlim(0, 255)

    ax2.plot(range(256), cdf)
    ax2.set_title(f"{title} – CDF")
    ax2.set_xlim(0, 255)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()


h1 = build_histogram([(0, 255)])                      # אחידה
h2 = build_histogram([(100, 150)])                    # מקטע אחד
h3 = build_histogram([(50, 90), (120, 180)])          # שני מקטעים
h4 = build_histogram([(50, 180)])                     # מסיח

plot_hist_and_cdf(h1, "Histogram 1")
plot_hist_and_cdf(h2, "Histogram 2")
plot_hist_and_cdf(h3, "Histogram 3")
plot_hist_and_cdf(h4, "Histogram 4 (Distractor)")
