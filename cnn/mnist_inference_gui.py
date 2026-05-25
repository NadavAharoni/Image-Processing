# ─────────────────────────────────────────────────────────────────────────────
# inference_gui.py
#
# Draw a digit with your mouse → the model classifies it in real time.
#
# Requirements: run train_and_save.py first to produce mnist_cnn.pth
# Dependencies: torch, torchvision, Pillow, tkinter (built into Python)
# ─────────────────────────────────────────────────────────────────────────────

import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageTk
from torchvision import transforms

SAVE_PATH     = "mnist_cnn.pth"
CANVAS_SIZE   = 280          # drawing canvas (pixels) — 10× the 28×28 input
BRUSH_RADIUS  = 14           # drawing brush size
PREVIEW_SCALE = 4            # display the 28×28 preview at 4× (112×112 px)
PREVIEW_DISP  = 28 * PREVIEW_SCALE
CONFIG_PATH   = Path.home() / ".mnist_gui_config.json"


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

        self._config        = self._load_config()
        self._last_probs    = [0.0] * 10
        self._last_digit    = None
        self._correct_label = tk.IntVar(value=-1)
        self._canvas_photo  = None  # prevent GC of canvas image on load
        self._preview_photo = None  # prevent GC of preview PhotoImage

        title_font = tkfont.Font(family="Helvetica", size=13, weight="bold")
        label_font = tkfont.Font(family="Helvetica", size=11)
        big_font   = tkfont.Font(family="Helvetica", size=48, weight="bold")
        bar_font   = tkfont.Font(family="Helvetica", size=9)

        # ── Left: drawing area ──
        left = tk.Frame(self, bg="#1e1e1e", padx=16, pady=16)
        left.grid(row=0, column=0, sticky="n")

        tk.Label(left, text="Draw a digit", font=title_font,
                 bg="#1e1e1e", fg="#cccccc").pack(anchor="w")
        tk.Label(left, text="(mouse to draw, right-click to clear)",
                 font=label_font, bg="#1e1e1e", fg="#666666").pack(anchor="w", pady=(0, 8))

        self.canvas = tk.Canvas(left, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg="black", cursor="crosshair",
                                highlightthickness=1, highlightbackground="#444")
        self.canvas.pack()

        # PIL image we draw into (same size as canvas, white-on-black like MNIST)
        self.pil_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.pil_draw  = ImageDraw.Draw(self.pil_image)

        self.canvas.bind("<B1-Motion>",       self._on_draw)
        self.canvas.bind("<Button-1>",        self._on_draw)
        self.canvas.bind("<Button-3>",        self._on_clear)   # right-click = clear
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        # 28×28 live preview
        tk.Label(left, text="28×28 model input", font=bar_font,
                 bg="#1e1e1e", fg="#666666").pack(anchor="w", pady=(10, 2))

        preview_border = tk.Frame(left, bg="#444444", padx=1, pady=1)
        preview_border.pack(anchor="w")

        self._preview_label = tk.Label(preview_border, bg="black")
        self._preview_label.pack()

        # Buttons row
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

        tk.Button(btn_frame, text="Load", font=label_font,
                  bg="#444444", fg="#cccccc", activebackground="#555",
                  relief="flat", padx=12, pady=6,
                  command=self._load_image).pack(side="left", padx=(8, 0))

        # ── Right: prediction panel ──
        right = tk.Frame(self, bg="#1e1e1e", padx=16, pady=16)
        right.grid(row=0, column=1, sticky="n", padx=(16, 0))

        tk.Label(right, text="Prediction", font=title_font,
                 bg="#1e1e1e", fg="#cccccc").pack(anchor="w")

        self.pred_label = tk.Label(right, text="?", font=big_font,
                                   bg="#1e1e1e", fg="#4a9eff", width=3)
        self.pred_label.pack(pady=(4, 12))

        tk.Label(right, text="Probabilities", font=label_font,
                 bg="#1e1e1e", fg="#888888").pack(anchor="w")

        # Bars + correct-label radio buttons in a shared grid for row alignment
        self.bars       = []
        self.pct_labels = []
        bars_radio = tk.Frame(right, bg="#1e1e1e")
        bars_radio.pack(anchor="w", pady=(4, 0))

        BAR_MAX = 180  # max bar width in pixels

        # "Label" header above the radio column
        tk.Label(bars_radio, text="Label", font=bar_font,
                 bg="#1e1e1e", fg="#888888").grid(row=0, column=3, padx=(32, 0), sticky="w")

        for digit in range(10):
            r = digit + 1  # row 0 is the header

            tk.Label(bars_radio, text=str(digit), font=bar_font, width=2,
                     bg="#1e1e1e", fg="#aaaaaa").grid(row=r, column=0, pady=1)

            bg_bar = tk.Frame(bars_radio, bg="#333333", width=BAR_MAX, height=14)
            bg_bar.grid(row=r, column=1, padx=(4, 4), pady=1)
            bg_bar.grid_propagate(False)

            fill = tk.Frame(bg_bar, bg="#4a9eff", width=0, height=14)
            fill.place(x=0, y=0, height=14)

            pct = tk.Label(bars_radio, text="  0%", font=bar_font, width=5,
                           bg="#1e1e1e", fg="#666666", anchor="w")
            pct.grid(row=r, column=2, pady=1)

            tk.Radiobutton(
                bars_radio, text=str(digit), variable=self._correct_label,
                value=digit, font=bar_font,
                bg="#1e1e1e", fg="#aaaaaa",
                activebackground="#1e1e1e", activeforeground="#cccccc",
                selectcolor="#1e1e1e",
                command=self._on_label_selected,
            ).grid(row=r, column=3, padx=(32, 0), pady=1, sticky="w")

            self.bars.append((fill, BAR_MAX))
            self.pct_labels.append(pct)

        # Save button — disabled until a correct label is chosen
        self._save_btn = tk.Button(
            right, text="Save", font=label_font,
            bg="#444444", fg="#888888", activebackground="#555",
            relief="flat", padx=12, pady=6,
            state="disabled",
            command=self._save_drawing,
        )
        self._save_btn.pack(anchor="w", pady=(10, 0))

        # Initialise preview with blank canvas
        self._update_preview()

    # ── Config ───────────────────────────────────────────────────────────────

    def _load_config(self):
        try:
            if CONFIG_PATH.exists():
                with open(CONFIG_PATH) as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _write_config(self):
        try:
            with open(CONFIG_PATH, "w") as f:
                json.dump(self._config, f)
        except Exception:
            pass

    def _default_save_dir(self):
        last = self._config.get("last_save_dir")
        if last and Path(last).is_dir():
            return str(last)
        return str(Path(__file__).parent / "images")

    # ── Preview ──────────────────────────────────────────────────────────────

    def _update_preview(self):
        # Resize to 28×28 with bilinear, matching transforms.Resize default
        small   = self.pil_image.resize((28, 28), Image.BILINEAR)
        # Scale up for display using nearest-neighbor for intentional pixelation
        display = small.resize((PREVIEW_DISP, PREVIEW_DISP), Image.NEAREST)
        photo   = ImageTk.PhotoImage(display)
        self._preview_label.config(image=photo)
        self._preview_photo = photo  # keep reference to prevent GC

    # ── Drawing ──────────────────────────────────────────────────────────────

    def _on_draw(self, event):
        x, y = event.x, event.y
        r = BRUSH_RADIUS
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
        self.pil_draw.ellipse([x-r, y-r, x+r, y+r], fill=255)

    def _on_release(self, event):
        # Auto-predict and update preview after each stroke
        self._run_predict()

    def _on_clear(self, event=None):
        self.canvas.delete("all")
        self.pil_draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)
        self.pred_label.config(text="?", fg="#4a9eff")
        for (fill, _), pct in zip(self.bars, self.pct_labels):
            fill.place(x=0, y=0, width=0, height=14)
            pct.config(text="  0%", fg="#666666")
        self._correct_label.set(-1)
        self._save_btn.config(state="disabled", fg="#888888", bg="#444444")
        self._last_probs = [0.0] * 10
        self._last_digit = None
        self._update_preview()

    # ── Inference ────────────────────────────────────────────────────────────

    def _run_predict(self):
        # Don't bother predicting on an empty canvas
        if self.pil_image.getextrema()[1] == 0:
            return

        digit, probs = predict(self.pil_image)
        self._last_digit = digit
        self._last_probs = probs
        self.pred_label.config(text=str(digit))
        self._update_preview()

        for i, (prob, (fill, bar_max), pct) in enumerate(
                zip(probs, self.bars, self.pct_labels)):
            w = int(prob * bar_max)
            color = "#4a9eff" if i == digit else "#555555"
            fill.place(x=0, y=0, width=w, height=14)
            fill.config(bg=color)
            pct.config(text=f"{prob*100:4.1f}%",
                       fg="#cccccc" if i == digit else "#666666")

    # ── Correct-label selector ────────────────────────────────────────────────

    def _on_label_selected(self):
        self._save_btn.config(state="normal", fg="white", bg="#4a7a4a",
                              activebackground="#5a8a5a")

    # ── Save ─────────────────────────────────────────────────────────────────

    def _next_index(self, directory):
        nums = []
        for f in Path(directory).glob("digit_*.png"):
            parts = f.stem.split("_")
            if len(parts) == 2:
                try:
                    nums.append(int(parts[1]))
                except ValueError:
                    pass
        return (max(nums) + 1) if nums else 1

    def _save_drawing(self):
        save_dir = self._default_save_dir()
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        nnn          = self._next_index(save_dir)
        default_name = f"digit_{nnn:03d}.png"

        filepath = filedialog.asksaveasfilename(
            initialdir=save_dir,
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[("PNG files", "*.png")],
            title="Save drawing",
        )
        if not filepath:
            return

        filepath = Path(filepath)
        self._config["last_save_dir"] = str(filepath.parent)
        self._write_config()

        stem      = filepath.stem
        json_path = filepath.parent / f"result_{stem}.json"

        try:
            small = self.pil_image.resize((28, 28), Image.BILINEAR)
            small.save(str(filepath))
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save image:\n{e}")
            return

        try:
            result = {
                "image_file":      filepath.name,
                "correct_label":   self._correct_label.get(),
                "predicted_label": self._last_digit,
                "probabilities":   {str(i): round(p, 4)
                                    for i, p in enumerate(self._last_probs)},
            }
            with open(json_path, "w") as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save results JSON:\n{e}")

    # ── Load ─────────────────────────────────────────────────────────────────

    def _load_image(self):
        load_dir = self._default_save_dir()

        filepath = filedialog.askopenfilename(
            initialdir=load_dir,
            filetypes=[("PNG files", "*.png")],
            title="Load drawing",
        )
        if not filepath:
            return

        try:
            img = Image.open(filepath).convert("L")
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load image:\n{e}")
            return

        self._config["last_save_dir"] = str(Path(filepath).parent)
        self._write_config()

        # Scale 28×28 up to canvas size using nearest-neighbor (no smoothing)
        img_canvas = img.resize((CANVAS_SIZE, CANVAS_SIZE), Image.NEAREST)

        self.pil_image = img_canvas.copy()
        self.pil_draw  = ImageDraw.Draw(self.pil_image)

        photo = ImageTk.PhotoImage(img_canvas)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=photo)
        self._canvas_photo = photo  # keep reference to prevent GC

        # Reset label selector and disable Save until user re-selects
        self._correct_label.set(-1)
        self._save_btn.config(state="disabled", fg="#888888", bg="#444444")

        # Run inference (also updates the preview)
        self._run_predict()
        # Ensure preview is shown even if the loaded image is blank
        self._update_preview()


if __name__ == "__main__":
    app = App()
    app.mainloop()
