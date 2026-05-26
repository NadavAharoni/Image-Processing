# ─────────────────────────────────────────────────────────────────────────────
# cifar_train_and_save.py
#
# Train a CNN on CIFAR-10 and save the weights to cifar_cnn.pth.
# Run this first, then open cifar_inference_gui.py to classify images.
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ── HYPERPARAMETERS ───────────────────────────────────────────────────────────

BLOCK1_FILTERS = 32     # filters in conv 1a and 1b
BLOCK2_FILTERS = 64     # filters in conv 2a and 2b
BLOCK3_FILTERS = 128    # filters in conv 3a and 3b
FC_HIDDEN       = 256   # neurons in the hidden FC layer

LEARNING_RATE   = 0.01
MOMENTUM        = 0.9   # used by SGD
WEIGHT_DECAY    = 1e-4  # L2 regularization
BATCH_SIZE      = 64
EPOCHS          = 30

SAVE_PATH       = "cifar_cnn.pth"

# ── TOGGLES ───────────────────────────────────────────────────────────────────
# Turn each technique on/off to observe its effect on accuracy.

USE_BATCH_NORM    = True   # normalize activations after each conv layer
USE_DROPOUT       = True   # randomly zero FC activations during training
USE_AUGMENTATION  = True   # random flips and crops during training
USE_LR_SCHEDULER  = True   # reduce learning rate when training plateaus

# ─────────────────────────────────────────────────────────────────────────────


# ── Data ──────────────────────────────────────────────────────────────────────

# CIFAR-10 channel means and stds (precomputed over the training set)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

if USE_AUGMENTATION:
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),             # a dog facing left is still a dog
        transforms.RandomCrop(32, padding=4),          # shift by up to 4 pixels
        transforms.ColorJitter(brightness=0.2,
                               contrast=0.2),          # slight colour variation
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
else:
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

train_data   = datasets.CIFAR10(root='./data', train=True,  download=True, transform=train_transform)
test_data    = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


# ── Model ─────────────────────────────────────────────────────────────────────

def conv_block(in_channels, out_channels, use_bn):
    """Two conv layers with optional batch norm, followed by max pool."""
    layers = []
    for in_ch in [in_channels, out_channels]:
        out_ch = out_channels
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)


class CifarCNN(nn.Module):
    def __init__(self, block1, block2, block3, fc_hidden, use_bn, use_dropout):
        super().__init__()

        # Three conv blocks: each halves spatial dims, doubles channels
        # Input:   3 × 32 × 32
        # Block 1: 32 × 16 × 16
        # Block 2: 64 × 8  × 8
        # Block 3: 128 × 4 × 4  →  flatten → 2048
        self.block1 = conv_block(3,     block1, use_bn)
        self.block2 = conv_block(block1, block2, use_bn)
        self.block3 = conv_block(block2, block3, use_bn)

        flat_size = block3 * 4 * 4  # after three 2×2 max pools on 32×32 input

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5) if use_dropout else nn.Identity(),
            nn.Linear(flat_size, fc_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.5) if use_dropout else nn.Identity(),
            nn.Linear(fc_hidden, 10)   # 10 output logits, one per class
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CifarCNN(
    block1=BLOCK1_FILTERS,
    block2=BLOCK2_FILTERS,
    block3=BLOCK3_FILTERS,
    fc_hidden=FC_HIDDEN,
    use_bn=USE_BATCH_NORM,
    use_dropout=USE_DROPOUT,
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model ready — {total_params:,} trainable parameters")
print(f"Toggles — batch norm: {USE_BATCH_NORM}  |  dropout: {USE_DROPOUT}  "
      f"|  augmentation: {USE_AUGMENTATION}  |  lr scheduler: {USE_LR_SCHEDULER}")
print(f"Filters: {BLOCK1_FILTERS} / {BLOCK2_FILTERS} / {BLOCK3_FILTERS}  "
      f"|  FC hidden: {FC_HIDDEN}\n")


# ── Training ──────────────────────────────────────────────────────────────────

# SGD with momentum generalises better than Adam on CIFAR-10
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,
                      momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

if USE_LR_SCHEDULER:
    # Halve the learning rate if val accuracy hasn't improved for 5 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5)

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
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch:02d}/{EPOCHS}  |  "
          f"train loss: {train_loss:.4f}  |  "
          f"test acc: {test_acc:.4f}  |  "
          f"lr: {current_lr:.5f}")

    if USE_LR_SCHEDULER:
        scheduler.step(test_acc)

# ── Save ──────────────────────────────────────────────────────────────────────

torch.save({
    'model_state': model.state_dict(),
    'config': {
        'block1_filters': BLOCK1_FILTERS,
        'block2_filters': BLOCK2_FILTERS,
        'block3_filters': BLOCK3_FILTERS,
        'fc_hidden':      FC_HIDDEN,
        'use_bn':         USE_BATCH_NORM,
        'use_dropout':    USE_DROPOUT,
    }
}, SAVE_PATH)

print(f"\nWeights saved to {SAVE_PATH}")
print("Open cifar_inference_gui.py to classify images.")
