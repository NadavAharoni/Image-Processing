# ─────────────────────────────────────────────────────────────────────────────
# cifar_inference_gui.py
#
# Classify CIFAR-10 images using the trained CNN.
# Two modes:
#   • Load any image file  → resized to 32×32, classified
#   • Browse test set      → cycle through CIFAR-10 test images
#
# Run train_and_save.py first to produce cifar_cnn.pth.
# ─────────────────────────────────────────────────────────────────────────────

import tkinter as tk
from tkinter import filedialog, font as tkfont
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image, ImageTk

SAVE_PATH   = "cifar_cnn.pth"
DISPLAY_SIZE = 256   # canvas display size (px) — image is upscaled from 32×32

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


# ── Model (must match cifar_train_and_save.py) ────────────────────────────────

def conv_block(in_channels, out_channels, use_bn):
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
        self.block1 = conv_block(3,      block1, use_bn)
        self.block2 = conv_block(block1, block2, use_bn)
        self.block3 = conv_block(block2, block3, use_bn)
        flat_size = block3 * 4 * 4
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5) if use_dropout else nn.Identity(),
            nn.Linear(flat_size, fc_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.5) if use_dropout else nn.Identity(),
            nn.Linear(fc_hidden, 10)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x)


# ── Load model ────────────────────────────────────────────────────────────────

checkpoint = torch.load(SAVE_PATH, map_location='cpu')
config     = checkpoint['config']
model      = CifarCNN(**config)
model.load_state_dict(checkpoint['model_state'])
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])


def predict(pil_image):
    """Return (predicted_class_index, list_of_10_probabilities)."""
    tensor = preprocess(pil_image.convert('RGB')).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
    probs = F.softmax(logits, dim=1).squeeze().tolist()
    return int(torch.argmax(logits)), probs


# ── Load CIFAR-10 test set (for browsing) ─────────────────────────────────────

# Raw test set — no normalisation — we want the original PIL images
raw_test = datasets.CIFAR10(root='./data', train=False, download=True,
                             transform=transforms.ToTensor())
test_index = 0   # current position in the test set


# ── GUI ───────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CIFAR-10 Classifier")
        self.resizable(False, False)
        self.configure(bg="#1e1e1e")

        title_font = tkfont.Font(family="Helvetica", size=13, weight="bold")
        label_font = tkfont.Font(family="Helvetica", size=11)
        big_font   = tkfont.Font(family="Helvetica", size=20, weight="bold")
        bar_font   = tkfont.Font(family="Helvetica", size=9)

        # ── Left: image display ───────────────────────────────────────────────
        left = tk.Frame(self, bg="#1e1e1e", padx=16, pady=16)
        left.grid(row=0, column=0, sticky="n")

        tk.Label(left, text="Image", font=title_font,
                 bg="#1e1e1e", fg="#cccccc").pack(anchor="w")
        tk.Label(left, text="32×32 input (upscaled for display)",
                 font=bar_font, bg="#1e1e1e", fg="#666666").pack(anchor="w", pady=(0, 8))

        self.canvas = tk.Label(left, width=DISPLAY_SIZE, height=DISPLAY_SIZE,
                               bg="#111111", relief="flat",
                               highlightthickness=1, highlightbackground="#444")
        self.canvas.pack()

        # True label (only shown when browsing test set)
        self.true_label_var = tk.StringVar(value="")
        self.true_label_lbl = tk.Label(left, textvariable=self.true_label_var,
                                       font=label_font, bg="#1e1e1e", fg="#888888")
        self.true_label_lbl.pack(pady=(6, 0))

        # Buttons
        btn_frame = tk.Frame(left, bg="#1e1e1e")
        btn_frame.pack(fill="x", pady=(10, 0))

        tk.Button(btn_frame, text="Load image", font=label_font,
                  bg="#4a9eff", fg="white", activebackground="#3a8eef",
                  relief="flat", padx=10, pady=6,
                  command=self._load_image).pack(side="left")

        tk.Button(btn_frame, text="◀ Prev", font=label_font,
                  bg="#444444", fg="#cccccc", activebackground="#555",
                  relief="flat", padx=10, pady=6,
                  command=self._prev_test).pack(side="left", padx=(8, 0))

        tk.Button(btn_frame, text="Next ▶", font=label_font,
                  bg="#444444", fg="#cccccc", activebackground="#555",
                  relief="flat", padx=10, pady=6,
                  command=self._next_test).pack(side="left", padx=(4, 0))

        # ── Right: prediction panel ───────────────────────────────────────────
        right = tk.Frame(self, bg="#1e1e1e", padx=16, pady=16)
        right.grid(row=0, column=1, sticky="n")

        tk.Label(right, text="Prediction", font=title_font,
                 bg="#1e1e1e", fg="#cccccc").pack(anchor="w")

        self.pred_label = tk.Label(right, text="—", font=big_font,
                                   bg="#1e1e1e", fg="#4a9eff", width=12,
                                   anchor="w")
        self.pred_label.pack(pady=(4, 12), anchor="w")

        tk.Label(right, text="Probabilities", font=label_font,
                 bg="#1e1e1e", fg="#888888").pack(anchor="w")

        self.bars       = []
        self.pct_labels = []
        BAR_MAX         = 180

        bar_frame = tk.Frame(right, bg="#1e1e1e")
        bar_frame.pack(fill="x", pady=(4, 0))

        for i, cls in enumerate(CLASSES):
            row = tk.Frame(bar_frame, bg="#1e1e1e")
            row.pack(fill="x", pady=1)

            tk.Label(row, text=cls, font=bar_font, width=10, anchor="w",
                     bg="#1e1e1e", fg="#aaaaaa").pack(side="left")

            bg_bar = tk.Frame(row, bg="#333333", width=BAR_MAX, height=14)
            bg_bar.pack(side="left", padx=(4, 4))
            bg_bar.pack_propagate(False)

            fill = tk.Frame(bg_bar, bg="#4a9eff", width=0, height=14)
            fill.place(x=0, y=0, height=14)

            pct = tk.Label(row, text="  0%", font=bar_font, width=5,
                           bg="#1e1e1e", fg="#666666", anchor="w")
            pct.pack(side="left")

            self.bars.append((fill, BAR_MAX))
            self.pct_labels.append(pct)

        self._BAR_MAX = BAR_MAX

        # Show the first test image on startup
        self._show_test_image(0)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _display_pil(self, pil_image):
        """Upscale a PIL image to DISPLAY_SIZE and show it on the canvas."""
        upscaled = pil_image.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.NEAREST)
        self._tk_image = ImageTk.PhotoImage(upscaled)
        self.canvas.configure(image=self._tk_image)

    def _update_results(self, pred_idx, probs, true_idx=None):
        """Update the prediction label and probability bars."""
        self.pred_label.config(text=CLASSES[pred_idx])

        if true_idx is not None:
            correct = pred_idx == true_idx
            status  = "✓ correct" if correct else f"✗ true: {CLASSES[true_idx]}"
            color   = "#4aff88" if correct else "#ff6b6b"
            self.true_label_var.set(status)
            self.true_label_lbl.config(fg=color)
        else:
            self.true_label_var.set("")

        for i, (prob, (fill, bar_max), pct) in enumerate(
                zip(probs, self.bars, self.pct_labels)):
            w     = int(prob * bar_max)
            color = "#4a9eff" if i == pred_idx else "#555555"
            fill.place(x=0, y=0, width=w, height=14)
            fill.config(bg=color)
            pct.config(text=f"{prob*100:4.1f}%",
                       fg="#cccccc" if i == pred_idx else "#666666")

    def _show_test_image(self, idx):
        """Display a CIFAR-10 test image by index and run inference."""
        global test_index
        test_index = idx % len(raw_test)

        tensor, true_label = raw_test[test_index]
        # tensor is C×H×W in [0,1] — convert to PIL for display and inference
        pil_image = transforms.ToPILImage()(tensor)

        self._display_pil(pil_image)
        pred_idx, probs = predict(pil_image)
        self._update_results(pred_idx, probs, true_idx=true_label)

    # ── Button callbacks ──────────────────────────────────────────────────────

    def _load_image(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All", "*.*")]
        )
        if not path:
            return
        pil_image = Image.open(path).convert('RGB')
        self._display_pil(pil_image)
        pred_idx, probs = predict(pil_image)
        self._update_results(pred_idx, probs, true_idx=None)

    def _next_test(self):
        self._show_test_image(test_index + 1)

    def _prev_test(self):
        self._show_test_image(test_index - 1)


if __name__ == "__main__":
    app = App()
    app.mainloop()
