import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# ── Configuration ─────────────────────────────────────────────────────────────
FEATURE_INDICES = [1, 4]   # mean texture, mean smoothness — change freely
LEARNING_RATE   = 0.1
EPOCHS          = 200
RANDOM_SEED     = 42

# ── Helper functions ──────────────────────────────────────────────────────────

def predict(X, layer):
    # nn.Linear computes X @ wᵀ + b internally — same math as before.
    # It holds the weights and bias for us; we no longer manage w and b ourselves.
    z = layer(X)              # shape: (n_samples, 1)
    return torch.sigmoid(z)   # same sigmoid as last week, just from PyTorch

# binary_cross_entropy is gone — nn.BCELoss() replaces it.
# It computes the exact same formula: -mean(y*log(ŷ) + (1-y)*log(1-ŷ))

# compute_gradients is gone — loss.backward() replaces it.
# See the training loop below for a detailed explanation.

def compute_accuracy(X, y_true, layer):
    with torch.no_grad():
        # torch.no_grad() tells PyTorch: "I'm not training right now,
        # so don't bother tracking gradients." Saves memory and speeds things up.
        y_pred = predict(X, layer)
        predicted_classes = (y_pred >= 0.5).float()
        return (predicted_classes.squeeze() == y_true).float().mean().item()
        # .item() converts a single-element tensor to a plain Python float

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(RANDOM_SEED)

    # ── 1. Load and prepare data ──────────────────────────────────────────────
    data          = load_breast_cancer()
    X_all         = data.data
    y_np          = data.target.astype(float)
    feature_names = data.feature_names

    X_np = X_all[:, FEATURE_INDICES]
    selected_names = [feature_names[i] for i in FEATURE_INDICES]
    n_features = X_np.shape[1]

    print(f"Features selected : {selected_names}")
    print(f"Dataset shape     : {X_np.shape}  (samples × features)")
    print(f"Class distribution: {int(y_np.sum())} benign, {int((1-y_np).sum())} malignant")

    # ── 2. Normalise ──────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_np   = scaler.fit_transform(X_np)

    # ── 3. Convert NumPy arrays to PyTorch tensors ────────────────────────────
    # PyTorch works with tensors, not NumPy arrays.
    # A tensor is essentially a NumPy array that PyTorch can differentiate through.
    # float32 is the standard precision for neural network training.
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np,  dtype=torch.float32)

    # ── 4. Define the model ───────────────────────────────────────────────────
    # nn.Linear(in, out) creates one layer with:
    #   - a weight matrix of shape (out, in)  — here (1, n_features)
    #   - a bias vector of shape (out,)        — here (1,)
    # It initialises them randomly (not zeros like we did last week).
    # This is our entire "model" — a single neuron, same as last week.
    layer = nn.Linear(in_features=n_features, out_features=1)

    # ── 5. Define the loss function ───────────────────────────────────────────
    # BCELoss = Binary Cross-Entropy Loss.
    # Computes the exact same formula as our binary_cross_entropy() from last week.
    loss_fn = nn.BCELoss()

    # ── 6. Define the optimiser ───────────────────────────────────────────────
    # The optimiser is responsible for updating the weights.
    # SGD (Stochastic Gradient Descent) does the same update rule we wrote manually:
    #   w = w - learning_rate * gradient
    # We hand it layer.parameters() so it knows which tensors to update.
    optimizer = torch.optim.SGD(layer.parameters(), lr=LEARNING_RATE)

    # ── 7. Training loop ──────────────────────────────────────────────────────
    loss_history = []

    for epoch in range(EPOCHS):

        # --- Forward pass: compute predictions ---
        y_pred = predict(X, layer)         # shape: (n_samples, 1)

        # y needs to match y_pred's shape for BCELoss
        loss = loss_fn(y_pred.squeeze(), y)

        # --- Backward pass: compute gradients ---
        # This is the key step that replaces our compute_gradients() function.
        #
        # loss.backward() does NOT update the weights.
        # It only COMPUTES the gradients (∂loss/∂w and ∂loss/∂b)
        # and stores them inside layer.weight.grad and layer.bias.grad.
        # Think of it as: "figure out which direction each weight should move."
        optimizer.zero_grad()   # clear gradients from the previous epoch first
        loss.backward()         # compute and store gradients

        # --- Weight update: apply the gradients ---
        # optimizer.step() is what actually CHANGES the weights, using:
        #   w = w - learning_rate * w.grad
        #   b = b - learning_rate * b.grad
        # This is exactly what we wrote manually last week.
        optimizer.step()

        loss_history.append(loss.item())

        if (epoch + 1) % 50 == 0:
            acc = compute_accuracy(X, y, layer)
            print(f"Epoch {epoch+1:4d} | loss: {loss.item():.4f} | accuracy: {acc:.3f}")

    final_acc = compute_accuracy(X, y, layer)
    print(f"\nFinal accuracy: {final_acc:.3f}")

    # layer.weight and layer.bias are tensors; .detach().numpy() converts them back
    # to NumPy for printing. .detach() is needed because they still "remember"
    # their gradient history — we just want the raw numbers here.
    w = layer.weight.detach().numpy().flatten()
    b = layer.bias.detach().numpy().item()
    print(f"Learned weights: {w.round(4)}")
    print(f"Learned bias   : {b:.4f}")

    # ── 8. Plot ───────────────────────────────────────────────────────────────
    # Identical to last week — convert tensors to NumPy for matplotlib
    X_np_plot = X.numpy()
    y_np_plot = y.numpy()

    two_features = (n_features == 2)

    if two_features:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(7, 5))
        axes = [axes]

    # --- Loss curve ---
    ax_loss = axes[-1]
    ax_loss.plot(loss_history, color='steelblue', linewidth=1.8)
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Binary cross-entropy loss')
    ax_loss.set_title('Training loss')
    ax_loss.grid(True, alpha=0.3)

    # --- Decision boundary ---
    if two_features:
        import numpy as np
        ax_sc     = axes[0]
        malignant = y_np_plot == 0
        benign    = y_np_plot == 1

        ax_sc.scatter(X_np_plot[malignant, 0], X_np_plot[malignant, 1],
                      label='Malignant', color='tomato',
                      alpha=0.6, edgecolors='k', linewidths=0.3)
        ax_sc.scatter(X_np_plot[benign, 0], X_np_plot[benign, 1],
                      label='Benign', color='steelblue',
                      alpha=0.6, edgecolors='k', linewidths=0.3)

        x1_range = np.linspace(X_np_plot[:, 0].min() - 0.5,
                               X_np_plot[:, 0].max() + 0.5, 200)
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
        f"Single-layer perceptron (PyTorch) — {', '.join(selected_names)}\n"
        f"lr={LEARNING_RATE}, epochs={EPOCHS}, accuracy={final_acc:.3f}",
        fontsize=11
    )
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
