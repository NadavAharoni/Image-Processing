# ─────────────────────────────────────────────────────────────────────────────
# cifar_network.py
#
# Layer definition for the CIFAR CNN, consumed by nn_diagram.py.
# ─────────────────────────────────────────────────────────────────────────────

NODES_PER_ROW = 4      # how many nodes before wrapping to next row

NETWORK = [
    {"id": "input",   "label": "Input",    "sublabel": "3×32×32",          "output": "",              "type": "input"},
    {"id": "conv1a",  "label": "Conv 1a",  "sublabel": "32 filters, 3×3, ReLU",  "output": "→ 32×32×32",   "type": "conv"},
    {"id": "conv1b",  "label": "Conv 1b",  "sublabel": "32 filters, 3×3, ReLU",  "output": "→ 32×32×32",   "type": "conv"},
    {"id": "pool1",   "label": "MaxPool",  "sublabel": "",                        "output": "→ 32×16×16",   "type": "pool"},
    {"id": "conv2a",  "label": "Conv 2a",  "sublabel": "64 filters, 3×3, ReLU",  "output": "→ 64×16×16",   "type": "conv"},
    {"id": "conv2b",  "label": "Conv 2b",  "sublabel": "64 filters, 3×3, ReLU",  "output": "→ 64×16×16",   "type": "conv"},
    {"id": "pool2",   "label": "MaxPool",  "sublabel": "",                        "output": "→ 64×8×8",     "type": "pool"},
    {"id": "conv3a",  "label": "Conv 3a",  "sublabel": "128 filters, 3×3, ReLU", "output": "→ 128×8×8",    "type": "conv"},
    {"id": "conv3b",  "label": "Conv 3b",  "sublabel": "128 filters, 3×3, ReLU", "output": "→ 128×8×8",    "type": "conv"},
    {"id": "pool3",   "label": "MaxPool",  "sublabel": "",                  "output": "→ 128×4×4",    "type": "pool"},
    {"id": "flatten", "label": "Flatten",  "sublabel": "",                  "output": "2048 units",    "type": "flatten"},
    {"id": "fc1",     "label": "FC layer", "sublabel": "2048 → 256",        "output": "ReLU → 256",    "type": "fc"},
    {"id": "fc2",     "label": "FC layer", "sublabel": "256 → 10",          "output": "",              "type": "fc"},
    {"id": "softmax", "label": "Softmax",  "sublabel": "10 classes",        "output": "",              "type": "output"},
]

# Edges: list of (from_id, to_id). Sequential by default — override here.
# Leave as None to auto-generate a simple chain from the NETWORK list order.
EDGES = None
