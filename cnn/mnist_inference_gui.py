# ─────────────────────────────────────────────────────────────────────────────
# inference_gui.py
#
# Draw a digit with your mouse → the model classifies it in real time.
#
# Requirements: run train_and_save.py first to produce mnist_cnn.pth
# Dependencies: torch, torchvision, Pillow, tkinter (built into Python)
# ─────────────────────────────────────────────────────────────────────────────

import tkinter as tk
from tkinter import font as tkfont
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw

SAVE_PATH    = "mnist_cnn.pth"
CANVAS_SIZE  = 280          # drawing canvas (pixels) — 10× the 28×28 input
BRUSH_RADIUS = 14           # drawing brush size


# ── Model (must match train_and_save.py) ──────────────────────────────────────

class MnistCNN(nn.Module):
    def __init__(self, conv1_filters, conv2_filters, fc_hidden):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, conv1_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(conv1_filters, conv2_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv2_filters * 7 * 7, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, 10)
        )

    def forward(self, x):
        return self.head(self.block2(self.block1(x)))


# ── Load model ────────────────────────────────────────────────────────────────

checkpoint = torch.load(SAVE_PATH, map_location='cpu')
config     = checkpoint['config']
model      = MnistCNN(**config)
model.load_state_dict(checkpoint['model_state'])
model.eval()

# Same normalisation used during training
preprocess = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def predict(pil_image):
    """Return (predicted_digit, list_of_10_probabilities)."""
    tensor = preprocess(pil_image).unsqueeze(0)   # add batch dim → 1×1×28×28
    with torch.no_grad():
        logits = model(tensor)
    probs = F.softmax(logits, dim=1).squeeze().tolist()
    return int(torch.argmax(logits)), probs


# ── GUI ───────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MNIST Classifier")
        self.resizable(False, False)
        self.configure(bg="#1e1e1e")

        title_font = tkfont.Font(family="Helvetica", size=13, weight="bold")
        label_font = tkfont.Font(family="Helvetica", size=11)
        big_font   = tkfont.Font(family="Helvetica", size=48, weight="bold")
        bar_font   = tkfont.Font(family="Helvetica", size=9)

        # ── Left: drawing area ──
        left = tk.Frame(self, bg="#1e1e1e", padx=16, pady=16)
        left.grid(row=0, column=0)

        tk.Label(left, text="Draw a digit", font=title_font,
                 bg="#1e1e1e", fg="#cccccc").pack(anchor="w")
        tk.Label(left, text="(mouse to draw, right-click to clear)",
                 font=label_font, bg="#1e1e1e", fg="#666666").pack(anchor="w", pady=(0,8))

        self.canvas = tk.Canvas(left, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg="black", cursor="crosshair",
                                highlightthickness=1, highlightbackground="#444")
        self.canvas.pack()

        # PIL image we draw into (same size as canvas, white-on-black like MNIST)
        self.pil_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.pil_draw  = ImageDraw.Draw(self.pil_image)

        self.canvas.bind("<B1-Motion>",    self._on_draw)
        self.canvas.bind("<Button-1>",     self._on_draw)
        self.canvas.bind("<Button-3>",     self._on_clear)   # right-click = clear
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        btn_frame = tk.Frame(left, bg="#1e1e1e")
        btn_frame.pack(fill="x", pady=(10, 0))

        tk.Button(btn_frame, text="Predict", font=label_font,
                  bg="#4a9eff", fg="white", activebackground="#3a8eef",
                  relief="flat", padx=12, pady=6,
                  command=self._run_predict).pack(side="left")

        tk.Button(btn_frame, text="Clear", font=label_font,
                  bg="#444444", fg="#cccccc", activebackground="#555",
                  relief="flat", padx=12, pady=6,
                  command=self._on_clear).pack(side="left", padx=(8, 0))

        # ── Right: prediction panel ──
        right = tk.Frame(self, bg="#1e1e1e", padx=16, pady=16)
        right.grid(row=0, column=1, sticky="n")

        tk.Label(right, text="Prediction", font=title_font,
                 bg="#1e1e1e", fg="#cccccc").pack(anchor="w")

        self.pred_label = tk.Label(right, text="?", font=big_font,
                                   bg="#1e1e1e", fg="#4a9eff", width=3)
        self.pred_label.pack(pady=(4, 12))

        tk.Label(right, text="Probabilities", font=label_font,
                 bg="#1e1e1e", fg="#888888").pack(anchor="w")

        # One bar per digit class
        self.bars   = []
        self.pct_labels = []
        bar_frame = tk.Frame(right, bg="#1e1e1e")
        bar_frame.pack(fill="x", pady=(4, 0))

        BAR_MAX = 180  # max bar width in pixels

        for digit in range(10):
            row = tk.Frame(bar_frame, bg="#1e1e1e")
            row.pack(fill="x", pady=1)

            tk.Label(row, text=str(digit), font=bar_font, width=2,
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

    # ── Drawing ──────────────────────────────────────────────────────────────

    def _on_draw(self, event):
        x, y = event.x, event.y
        r = BRUSH_RADIUS
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
        self.pil_draw.ellipse([x-r, y-r, x+r, y+r], fill=255)

    def _on_release(self, event):
        # Auto-predict after each stroke so the bars update live
        self._run_predict()

    def _on_clear(self, event=None):
        self.canvas.delete("all")
        self.pil_draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)
        self.pred_label.config(text="?", fg="#4a9eff")
        for (fill, _), pct in zip(self.bars, self.pct_labels):
            fill.place(x=0, y=0, width=0, height=14)
            pct.config(text="  0%", fg="#666666")

    # ── Inference ────────────────────────────────────────────────────────────

    def _run_predict(self):
        # Don't bother predicting on an empty canvas
        if self.pil_image.getextrema()[1] == 0:
            return

        digit, probs = predict(self.pil_image)
        self.pred_label.config(text=str(digit))

        for i, (prob, (fill, bar_max), pct) in enumerate(
                zip(probs, self.bars, self.pct_labels)):
            w = int(prob * bar_max)
            color = "#4a9eff" if i == digit else "#555555"
            fill.place(x=0, y=0, width=w, height=14)
            fill.config(bg=color)
            pct.config(text=f"{prob*100:4.1f}%",
                       fg="#cccccc" if i == digit else "#666666")


if __name__ == "__main__":
    app = App()
    app.mainloop()
