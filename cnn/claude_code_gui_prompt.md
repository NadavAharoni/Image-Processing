# Claude Code Prompt — Extend `mnist_inference_gui.py`

## Context

This is a Tkinter-based GUI that lets a user draw a digit on a canvas,
runs it through a trained MNIST CNN, and displays the predicted class
probabilities as a bar chart. The script lives alongside `mnist_cnn.pth`
(the saved model weights).

## Goal

Add three capabilities:
1. **Save** the current drawing as a 28×28 PNG + a JSON results file
2. **Load** a previously saved 28×28 PNG back into the canvas and re-run inference
3. **Label selector** so the user can record the correct digit before saving

---

## Detailed requirements

### 1. 28×28 live preview

- Add a small panel to the GUI showing the 28×28 version of whatever is
  currently on the canvas. This is exactly the image that gets fed into
  the model after `ToTensor` but **before** `Normalize`.
- Update this preview in real time after every brush stroke — same
  cadence as the probability bars currently update.
- When a saved image is loaded (see §3), the canvas should display the
  image scaled back up to canvas size using **nearest-neighbor
  interpolation** (no smoothing). The 28×28 preview should show the same
  image. The pixelated appearance is intentional and desirable.
- There should be only one image visible at a time — no separate
  "original drawing" vs "model input" panels.

---

### 2. Correct-label selector

- Add a **column** of 10 radio buttons labeled 0–9, placed to the right
  of the probability bars in the results panel, with each radio button
  aligned to its corresponding bar row. Use a shared grid so the rows
  stay in sync. Add a small "Label" header above the column.
- Leave ~32 px of horizontal gap between the bar area and the radio
  column (via `padx` on the grid cell). Add extra left padding to the
  entire results panel so it is visually separated from the drawing
  canvas.
- The user selects the digit they intended to draw. This is the ground
  truth label.
- The **Save button** (see §3) must remain **disabled** until a radio
  button is selected. Re-enable it as soon as a selection is made.
- When the canvas is cleared, reset the radio button selection and
  disable the Save button again.

---

### 3. Save drawing + results

#### Trigger
A **Save** button in the GUI. Disabled until a correct label is selected
(see §2). Clicking it opens a file save dialog pre-filled with a default
path and filename (see below), then saves two files.

#### Default save directory
- `./images/` relative to the directory where the Python script lives
  (i.e. `Path(__file__).parent / "images"`).
- Create this directory if it does not exist.
- **Persist the last-used directory** across sessions using a config file
  at `~/.mnist_gui_config.json`. On startup, read this file and use its
  `last_save_dir` value as the default if present. On save, write the
  chosen directory back to this file.

#### Default filename
- Pattern: `digit_NNN.png` where `NNN` is a zero-padded 3-digit integer.
- Determine `NNN` by scanning the target directory for existing files
  matching `digit_*.png`, finding the highest number, and incrementing
  by one. Start at `001` if none exist.

#### Files saved (using the same `NNN`)
1. **`digit_NNN.png`** — the raw 28×28 grayscale image, pixel values
   0–255, standard PNG. This must be viewable in any image viewer and
   must be reloadable by this GUI without any preprocessing.
2. **`result_NNN.json`** — inference results, with this structure:

```json
{
  "image_file": "digit_001.png",
  "correct_label": 8,
  "predicted_label": 3,
  "probabilities": {
    "0": 0.0012,
    "1": 0.0003,
    "2": 0.0187,
    "3": 0.6201,
    "4": 0.0008,
    "5": 0.0234,
    "6": 0.0011,
    "7": 0.0009,
    "8": 0.3301,
    "9": 0.0034
  }
}
```

---

### 4. Load image

#### Trigger
A **Load** button in the GUI. Clicking it opens a file picker dialog
filtered to `.png` files. The initial directory should follow the same
logic as Save (last-used directory from config, falling back to
`./images/`).

#### Behavior after loading
- The selected PNG is expected to be a 28×28 grayscale image (as saved
  by this GUI). Read it as-is — do **not** resize or alter pixel values.
- Scale it up to the canvas size using nearest-neighbor interpolation and
  display it on the canvas.
- Update the 28×28 preview panel.
- Run inference immediately using the same preprocessing path as a fresh
  drawing: `ToTensor` → `Normalize((0.1307,), (0.3081,))`.
- Update the probability bars and predicted label display.
- Reset the correct-label radio buttons and disable the Save button
  (the user must re-select a label before saving).
- Update the last-used directory in `~/.mnist_gui_config.json`.

---

## Constraints

- Do not change the model class definition, model loading code, or
  inference logic.
- Do not change the drawing, brush, or canvas-clear logic except as
  needed to hook in the 28×28 preview update.
- Do not change the hyperparameter section or any training-related code
  (there is none in this file, but do not add any).
- Use only the Python standard library plus the packages already imported
  in the file (`torch`, `torchvision`, `Pillow`, `tkinter`). Do not add
  new dependencies.
- The save dialog should default to the computed `digit_NNN.png` filename
  but allow the user to override it. If the user changes the filename,
  derive the JSON filename from whatever the user chose (same base name,
  `result_` prefix, `.json` extension).
- All file I/O should be wrapped in try/except with a clear error message
  shown in the GUI (a `tkinter.messagebox.showerror` dialog is fine).

---

## Open for later (do not implement now)

- Model versioning: a `model_version` field in the JSON. This will be
  added in a follow-up step once a versioning scheme is decided.
