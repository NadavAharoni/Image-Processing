"""
wedge_dataset.py
----------------
Generates a synthetic 2D classification dataset where class 1 is defined by
a wedge region — the intersection of two positive half-planes. This is
designed to demonstrate that a single linear layer cannot separate the classes,
but a two-neuron ReLU hidden layer can.

Each line is defined as:
    f_i(x, y) = a_i*x + b_i*y + c_i

Class 1 is the region where BOTH f_1 > 0 AND f_2 > 0.
Class 0 is everywhere else (the other three wedge regions).

The probability of a point being assigned class 1 is based on how deep it sits
inside the correct region, using a sigmoid of the minimum signed distance to
the two lines. Points near either line are ambiguous; points deep in the wedge
are almost certainly class 1.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ---------------------------------------------------------------------------
# Line definitions
# ---------------------------------------------------------------------------

def signed_distance(x, y, a, b, c):
    """
    Compute the signed distance from point (x, y) to the line ax + by + c = 0.

    The sign indicates which half-plane the point is in:
      - Positive: same side as the normal vector (a, b)
      - Negative: opposite side

    Parameters
    ----------
    x, y : float or np.ndarray
        Coordinates of the point(s).
    a, b, c : float
        Line coefficients (ax + by + c = 0).

    Returns
    -------
    float or np.ndarray
        Signed distance(s).
    """
    norm = np.sqrt(a**2 + b**2)
    return (a * x + b * y + c) / norm


def class1_probability(x, y, a1, b1, c1, a2, b2, c2, steepness=5.0):
    """
    Compute the probability that point (x, y) belongs to class 1.

    Class 1 is defined as the wedge where both signed distances are positive.
    The probability is a sigmoid of the minimum of the two signed distances,
    scaled by `steepness`. This means:
      - Points deep inside the wedge (both distances large and positive) → prob ≈ 1
      - Points near either boundary line → prob ≈ 0.5
      - Points outside the wedge → prob < 0.5

    Parameters
    ----------
    x, y : float or np.ndarray
        Coordinates of the point(s).
    a1, b1, c1 : float
        Coefficients of the first line.
    a2, b2, c2 : float
        Coefficients of the second line.
    steepness : float
        Controls how sharply probability transitions near the boundary.
        Higher values = harder boundary.

    Returns
    -------
    float or np.ndarray
        Probability of class 1, in [0, 1].
    """
    d1 = signed_distance(x, y, a1, b1, c1)
    d2 = signed_distance(x, y, a2, b2, c2)
    # Use the minimum: a point must be inside BOTH half-planes to be class 1
    min_dist = np.minimum(d1, d2)
    return 1.0 / (1.0 + np.exp(-steepness * min_dist))


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def find_intersection(a1, b1, c1, a2, b2, c2):
    """
    Find the intersection point of two lines ax + by + c = 0.

    Parameters
    ----------
    a1, b1, c1 : float
        Coefficients of the first line.
    a2, b2, c2 : float
        Coefficients of the second line.

    Returns
    -------
    (float, float)
        (x, y) coordinates of the intersection, or None if lines are parallel.
    """
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-10:
        return None  # Lines are parallel
    x = (-c1 * b2 + c2 * b1) / det
    y = (-a1 * c2 + a2 * c1) / det
    return x, y


def generate_wedge_dataset(
    a1=1.0, b1=1.0, c1=-1.0,
    a2=1.0, b2=-1.0, c2=-1.0,
    n_samples=400,
    radius=3.0,
    steepness=5.0,
    random_seed=42
):
    """
    Generate a 2D wedge dataset around the intersection of two lines.

    Points are sampled uniformly within a circle of the given radius centered
    at the intersection of the two lines. Each point is assigned a class
    probabilistically based on how deep it sits in the class-1 wedge region.

    Parameters
    ----------
    a1, b1, c1 : float
        Coefficients of the first boundary line (a1*x + b1*y + c1 = 0).
    a2, b2, c2 : float
        Coefficients of the second boundary line (a2*x + b2*y + c2 = 0).
    n_samples : int
        Total number of points to generate.
    radius : float
        Radius of the circular sampling region around the intersection.
    steepness : float
        Controls sharpness of the probabilistic boundary (passed to
        class1_probability).
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray of shape (n_samples, 2)
        The 2D coordinates of each point.
    y : np.ndarray of shape (n_samples,)
        Binary class labels (0 or 1).
    intersection : (float, float) or None
        The intersection point of the two lines.
    """
    rng = np.random.default_rng(random_seed)

    # Find intersection to center the sampling region
    intersection = find_intersection(a1, b1, c1, a2, b2, c2)
    if intersection is None:
        raise ValueError("The two lines are parallel and do not intersect.")
    cx, cy = intersection

    # Sample points uniformly inside a circle using rejection sampling
    points = []
    while len(points) < n_samples:
        # Sample in bounding square, reject outside circle
        batch = rng.uniform(-radius, radius, size=(n_samples * 4, 2))
        inside = np.sum(batch**2, axis=1) <= radius**2
        batch = batch[inside]
        points.append(batch)

    X_centered = np.vstack(points)[:n_samples]

    # Shift to intersection point
    X = X_centered + np.array([cx, cy])

    # Assign labels probabilistically
    probs = class1_probability(
        X[:, 0], X[:, 1],
        a1, b1, c1,
        a2, b2, c2,
        steepness=steepness
    )
    y = (rng.uniform(size=n_samples) < probs).astype(int)

    return X, y, intersection


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_wedge_dataset(X, y, a1, b1, c1, a2, b2, c2, intersection=None):
    """
    Visualize the wedge dataset with the two boundary lines.

    Points are colored by class. The two boundary lines are drawn across the
    plot range. The intersection point is marked if provided.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, 2)
        Point coordinates.
    y : np.ndarray of shape (n_samples,)
        Binary class labels.
    a1, b1, c1 : float
        Coefficients of the first boundary line.
    a2, b2, c2 : float
        Coefficients of the second boundary line.
    intersection : (float, float) or None
        Intersection point to mark on the plot.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    # Scatter plot by class
    ax.scatter(X[y == 0, 0], X[y == 0, 1],
               color='steelblue', alpha=0.6, s=25, label='Class 0')
    ax.scatter(X[y == 1, 0], X[y == 1, 1],
               color='salmon', alpha=0.6, s=25, label='Class 1 (wedge)')

    # Draw boundary lines across the plot range
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    # x_range = np.linspace(x_min, x_max, 200)

    # def line_y(x, a, b, c):
    #     """Compute y from ax + by + c = 0 → y = -(ax + c) / b"""
    #     if abs(b) < 1e-10:
    #         return None  # Vertical line
    #     return -(a * x + c) / b

    # y1 = line_y(x_range, a1, b1, c1)
    # y2 = line_y(x_range, a2, b2, c2)

    # if y1 is not None:
    #     ax.plot(x_range, y1, color='darkgreen', linewidth=2,
    #             linestyle='--', label='Line 1')
    # if y2 is not None:
    #     ax.plot(x_range, y2, color='olive', linewidth=2,
    #             linestyle='--', label='Line 2')

    # Mark intersection
    if intersection is not None:
        ax.scatter(*intersection, color='black', s=80, zorder=5,
                   label=f'Intersection {intersection[0]:.1f}, {intersection[1]:.1f}')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title('Wedge Dataset — requires two separating planes')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('wedge_dataset.png', dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Generate and visualize the wedge dataset.

    Line parameters are defined here as constants. To change the wedge shape,
    adjust a1, b1, c1, a2, b2, c2. The class-1 wedge is where both
    a_i*x + b_i*y + c_i > 0.

    Default lines intersect at the origin, opening into the upper-right wedge:
      Line 1: x + y - 0  = 0  (diagonal, northeast normal)
      Line 2: x - y - 0  = 0  (diagonal, southeast normal)
    Together these define a wedge pointing right (positive x direction).
    """
    # --- Line 1: a1*x + b1*y + c1 = 0 ---
    a1, b1, c1 = 1.0, 1.0, 0.0   # x + y = 0, normal points up-right

    # --- Line 2: a2*x + b2*y + c2 = 0 ---
    a2, b2, c2 = 1.0, -1.0, 0.0  # x - y = 0, normal points down-right

    # --- Dataset parameters ---
    n_samples = 500
    radius = 3.0       # Sampling radius around intersection
    steepness = 3.0    # Sharpness of probabilistic boundary
    random_seed = 42

    # Generate
    X, y, intersection = generate_wedge_dataset(
        a1, b1, c1,
        a2, b2, c2,
        n_samples=n_samples,
        radius=radius,
        steepness=steepness,
        random_seed=random_seed
    )

    print(f"Generated {len(X)} points")
    print(f"Class 0: {(y == 0).sum()}  Class 1: {(y == 1).sum()}")
    print(f"Intersection at: {intersection}")

    # Visualize
    plot_wedge_dataset(X, y, a1, b1, c1, a2, b2, c2, intersection)


if __name__ == '__main__':
    main()
