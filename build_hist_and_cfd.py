import numpy as np
import matplotlib.pyplot as plt


def build_histogram(ranges, default_value=10):
    """
    ranges: list of (start, end) inclusive
    value: frequency for each bin in range
    """
    hist = np.zeros(256, dtype=int)

    for r in ranges:
        if len(r) == 2:
            start, end = r
            value = default_value
        else:
            start, end, value = r

        hist[start:end + 1] = value

    return hist


def compute_cdf(hist):
    cdf = np.cumsum(hist)
    if cdf.max() > 0:
        cdf = cdf / cdf.max()
    return cdf


def plot_hist_and_cdf(hist, title):
    cdf = compute_cdf(hist)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6))

    x = np.arange(256)

    # צבעים מתחלפים בגווני אפור (ידידותי להדפסה)
    bar_colors = ['#bbbbbb' if i % 2 == 0 else '#777777' for i in range(256)]

    ax1.bar(x, hist, width=1.0, color=bar_colors, edgecolor='black', linewidth=0.3)
    ax1.set_title(f"{title} – Histogram")
    ax1.set_xlim(0, 255)
    ax1.set_ylabel("Frequency")

    ax2.plot(x, cdf, color='black', linewidth=2)
    ax2.set_title(f"{title} – CDF")
    ax2.set_xlim(0, 255)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Gray Level")
    ax2.set_ylabel("CDF")

    xticks = np.arange(0, 256, 25)
    ax1.set_xticks(xticks)
    ax2.set_xticks(xticks)

    plt.tight_layout()
    plt.show()


def main():
    # היסטוגרמות "רגילות"
    h1 = build_histogram([(0, 255)])                      # אחידה 
    h2 = build_histogram([(100, 150)])                    # מקטע אחד
    h3 = build_histogram([(50, 90), (120, 180)])          # שני מקטעים
    h4 = build_histogram([(50, 180)])                     # רציף 

    # היסטוגרמה מדורגת – החדשה
    h5 = build_histogram([
        (50, 100, 5),
        (101, 150, 10),
        (151, 180, 20)
    ])

    h6 = build_histogram([
        (50, 100, 20),
        (101, 150, 10),
        (151, 180, 5)
    ])

    h7 = build_histogram([
        (50, 90, 10),
        (120, 180,3)
    ])

    plot_hist_and_cdf(h1, "Histogram 1")
    plot_hist_and_cdf(h2, "Histogram 2")
    plot_hist_and_cdf(h3, "Histogram 3")
    plot_hist_and_cdf(h4, "Histogram 4")
    plot_hist_and_cdf(h5, "Histogram 5")
    plot_hist_and_cdf(h6, "Histogram 6")
    plot_hist_and_cdf(h7, "Histogram 7")

if __name__ == "__main__":
    main()
