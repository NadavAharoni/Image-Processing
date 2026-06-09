import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# ── Configuration ─────────────────────────────────────────────────────────────
FEATURE_INDICES = [1, 4]   # mean texture, mean smoothness — change freely
LEARNING_RATE   = 0.1
EPOCHS          = 200
RANDOM_SEED     = 42

# ── Helper functions ──────────────────────────────────────────────────────────

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)  # avoid log(0)
    return -np.mean(
        y_true * np.log(y_pred) +
        (1 - y_true) * np.log(1 - y_pred)
    )

def predict(X, w, b):
    z = X @ w + b          # shape: (n_samples,)
    return sigmoid(z)

def compute_gradients(X, y_true, y_pred):
    error = y_pred - y_true              # (ŷ - y), shape: (n_samples,)
    dw = X.T @ error / len(y_true)       # shape: (n_features,)
    db = np.mean(error)                  # scalar
    return dw, db

def compute_accuracy(X, y_true, w, b):
    y_pred = predict(X, w, b)
    predicted_classes = (y_pred >= 0.5).astype(int)
    return np.mean(predicted_classes == y_true)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(RANDOM_SEED)

    # ── 1. Load and prepare data ──────────────────────────────────────────────
    data         = load_breast_cancer()
    X_all        = data.data
    y            = data.target.astype(float)
    feature_names = data.feature_names

    X = X_all[:, FEATURE_INDICES]
    selected_names = [feature_names[i] for i in FEATURE_INDICES]
    n_features = X.shape[1]

    print(f"Features selected : {selected_names}")
    print(f"Dataset shape     : {X.shape}  (samples × features)")
    print(f"Class distribution: {int(y.sum())} benign, {int((1-y).sum())} malignant")

    # ── 2. Normalise ──────────────────────────────────────────────────────────
    # Gradient descent is sensitive to feature scale.
    # StandardScaler gives each feature mean=0, std=1.
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # ── 3. Initialise weights ─────────────────────────────────────────────────
    w = np.zeros(n_features)
    b = 0.0

    # ── 4. Training loop ──────────────────────────────────────────────────────
    loss_history = []

    for epoch in range(EPOCHS):
        y_pred = predict(X, w, b)
        loss   = binary_cross_entropy(y, y_pred)
        loss_history.append(loss)

        dw, db = compute_gradients(X, y, y_pred)
        w -= LEARNING_RATE * dw
        b -= LEARNING_RATE * db

        if (epoch + 1) % 50 == 0:
            acc = compute_accuracy(X, y, w, b)
            print(f"Epoch {epoch+1:4d} | loss: {loss:.4f} | accuracy: {acc:.3f}")

    final_acc = compute_accuracy(X, y, w, b)
    print(f"\nFinal accuracy: {final_acc:.3f}")
    print(f"Learned weights: {np.round(w, 4)}")
    print(f"Learned bias   : {b:.4f}")

    # ── 5. Plot ───────────────────────────────────────────────────────────────
    two_features = (n_features == 2)

    if two_features:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(7, 5))
        axes = [axes]   # keep indexing consistent

    # --- Loss curve (always shown) ---
    ax_loss = axes[-1]
    ax_loss.plot(loss_history, color='steelblue', linewidth=1.8)
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Binary cross-entropy loss')
    ax_loss.set_title('Training loss')
    ax_loss.grid(True, alpha=0.3)

    # --- Decision boundary (only when exactly 2 features) ---
    if two_features:
        ax_sc = axes[0]
        malignant = y == 0
        benign    = y == 1

        ax_sc.scatter(X[malignant, 0], X[malignant, 1],
                      label='Malignant', color='tomato',
                      alpha=0.6, edgecolors='k', linewidths=0.3)
        ax_sc.scatter(X[benign, 0], X[benign, 1],
                      label='Benign', color='steelblue',
                      alpha=0.6, edgecolors='k', linewidths=0.3)

        # Decision boundary: w[0]*x1 + w[1]*x2 + b = 0  →  x2 = -(w[0]*x1 + b) / w[1]
        x1_range = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200)
        if abs(w[1]) > 1e-8:
            x2_boundary = -(w[0] * x1_range + b) / w[1]
            ax_sc.plot(x1_range, x2_boundary,
                       color='black', linewidth=1.5,
                       linestyle='--', label='Decision boundary')

        ax_sc.set_xlabel(f'{selected_names[0]} (normalised)')
        ax_sc.set_ylabel(f'{selected_names[1]} (normalised)')
        ax_sc.set_title('Data + learned decision boundary')
        ax_sc.legend()
        ax_sc.grid(True, alpha=0.3)

    plt.suptitle(
        f"Single-layer perceptron — {', '.join(selected_names)}\n"
        f"lr={LEARNING_RATE}, epochs={EPOCHS}, accuracy={final_acc:.3f}",
        fontsize=11
    )
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
