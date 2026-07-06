# ─────────────────────────────────────────────────────────────────────────────
# mnist_network.py
#
# Layer definition for the MNIST CNN, consumed by nn_diagram.py.
# ─────────────────────────────────────────────────────────────────────────────

NODES_PER_ROW = 5      # how many nodes before wrapping to next row

NETWORK = [
    {"id": "input",   "label": "Input",    "sublabel": "1×28×28",           "output": "",              "type": "input"},
    {"id": "conv1",   "label": "Conv 1",   "sublabel": "8 filters, 3×3, ReLU",   "output": "→ 8×28×28",    "type": "conv"},
    {"id": "pool1",   "label": "MaxPool",  "sublabel": "",                        "output": "→ 8×14×14",   "type": "pool"},
    {"id": "conv2",   "label": "Conv 2",   "sublabel": "16 filters, 3×3, ReLU",  "output": "→ 16×14×14",  "type": "conv"},
    {"id": "pool2",   "label": "MaxPool",  "sublabel": "",                        "output": "→ 16×7×7",    "type": "pool"},
    {"id": "flatten", "label": "Flatten",  "sublabel": "",                  "output": "784 units",     "type": "flatten"},
    {"id": "fc1",     "label": "FC layer", "sublabel": "784 → 128, ReLU",   "output": "ReLU → 128",    "type": "fc"},
    {"id": "fc2",     "label": "FC layer", "sublabel": "128 → 10",          "output": "",              "type": "fc"},
    {"id": "softmax", "label": "Softmax",  "sublabel": "10 classes",        "output": "",              "type": "output"},
]

# Edges: list of (from_id, to_id). Sequential by default — override here.
# Leave as None to auto-generate a simple chain from the NETWORK list order.
EDGES = None
