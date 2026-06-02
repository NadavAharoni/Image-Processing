# ─────────────────────────────────────────────────────────────────────────────
# MNIST CNN — student playground
#
# The only section you need to edit is "HYPERPARAMETERS" below.
# Everything else (data loading, training loop, evaluation) stays the same.
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ── Argument parsing ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--augment", action="store_true",
                    help="Apply data augmentation to training set")
args = parser.parse_args()

SAVE_PATH = "mnist_cnn_augmented.pth" if args.augment else "mnist_cnn.pth"

# ── HYPERPARAMETERS ───────────────────────────────────────────────────────────

CONV1_FILTERS = 8       # filters in the first conv layer  (try: 2, 4, 8, 16, 32)
CONV2_FILTERS = 16      # filters in the second conv layer (try: 8, 16, 32, 64)
FC_HIDDEN     = 128     # neurons in the hidden FC layer   (try: 32, 64, 128, 256)

LEARNING_RATE = 0.001
BATCH_SIZE    = 64
EPOCHS        = 5

# ─────────────────────────────────────────────────────────────────────────────


# ── Data ──────────────────────────────────────────────────────────────────────

base_train_transforms = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
if args.augment:
    base_train_transforms = [
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    ] + base_train_transforms
train_transform = transforms.Compose(base_train_transforms)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST(root='./data', train=True,  download=True, transform=train_transform)
test_data  = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)


# ── Model ─────────────────────────────────────────────────────────────────────

class MnistCNN(nn.Module):
    def __init__(self, conv1_filters, conv2_filters, fc_hidden):
        super().__init__()

        # Block 1: conv → ReLU → maxpool
        # Input:  1 × 28 × 28
        # Output: conv1_filters × 14 × 14
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=conv1_filters,
                      kernel_size=3, padding=1),  # same padding keeps 28×28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # halves spatial dims → 14×14
        )

        # Block 2: conv → ReLU → maxpool
        # Input:  conv1_filters × 14 × 14
        # Output: conv2_filters × 7 × 7
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=conv1_filters, out_channels=conv2_filters,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # → 7×7
        )

        # After block2, the feature map is conv2_filters × 7 × 7
        # We flatten it to a vector of size conv2_filters * 7 * 7
        flat_size = conv2_filters * 7 * 7

        # Fully connected head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, 10)  # 10 output logits, one per digit class
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.head(x)
        return x  # raw logits (CrossEntropyLoss expects these, not softmax)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MnistCNN(
    conv1_filters=CONV1_FILTERS,
    conv2_filters=CONV2_FILTERS,
    fc_hidden=FC_HIDDEN
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model ready — {total_params:,} trainable parameters")
print(f"Conv filters: {CONV1_FILTERS} → {CONV2_FILTERS}  |  FC hidden: {FC_HIDDEN}\n")


# ── Training loop ─────────────────────────────────────────────────────────────

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"Augmentation: {'ON' if args.augment else 'OFF'} — saving to {SAVE_PATH}")

for epoch in range(1, EPOCHS + 1):
    # ── train ──
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    train_loss /= len(train_loader.dataset)

    # ── evaluate ──
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print(f"Epoch {epoch}/{EPOCHS}  |  train loss: {train_loss:.4f}  |  test acc: {test_acc:.4f}")

torch.save({
    'model_state': model.state_dict(),
    'config': {
        'conv1_filters': CONV1_FILTERS,
        'conv2_filters': CONV2_FILTERS,
        'fc_hidden':     FC_HIDDEN,
    }
}, SAVE_PATH)

print(f"\nWeights saved to {SAVE_PATH}")
print("Open inference_gui.py to classify digits you draw.")
