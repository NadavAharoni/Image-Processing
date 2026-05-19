import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ── 1. Data ────────────────────────────────────────────────────────────────────
X_raw, y_raw = load_iris(return_X_y=True)          # (150, 4), labels in {0,1,2}

X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)            # fit on train only
X_test  = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=16, shuffle=True
)

# ── 2. Model ───────────────────────────────────────────────────────────────────
#
#   Input (4 features)
#       ↓
#   Linear(4 → 3)        ← single layer: weight matrix is (4 × 3)
#       ↓
#   (CrossEntropyLoss applies softmax internally)
#
class IrisNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 3)   # one score per class, no hidden layer

    def forward(self, x):
        return self.layer(x)           # raw logits; shape (batch, 3)

model = IrisNet()

# ── 3. Training ────────────────────────────────────────────────────────────────
# CrossEntropyLoss = softmax + negative log-likelihood combined
loss_fn   = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

EPOCHS = 100
train_losses = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()           # clear old gradients
        logits = model(X_batch)         # forward pass  → (batch, 3)
        loss   = loss_fn(logits, y_batch)
        loss.backward()                 # backprop
        optimizer.step()                # update weights
        epoch_loss += loss.item() * len(X_batch)

    train_losses.append(epoch_loss / len(X_train_t))

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d} | loss: {train_losses[-1]:.4f}")

# ── 4. Evaluation ──────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    logits = model(X_test_t)            # (30, 3)
    preds  = logits.argmax(dim=1)       # class with highest score
    acc    = (preds == y_test_t).float().mean().item()

print(f"\nTest accuracy: {acc*100:.1f}%")

# ── 5. Loss curve ──────────────────────────────────────────────────────────────
plt.figure(figsize=(6, 3))
plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.title("Training loss — Iris single-layer classifier")
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=120)
plt.show()
