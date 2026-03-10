import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk

import warp

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Affine Transform Demo")
        self.root.geometry("600x500")

        # ---- State ----
        self.original_image = None      # Original BGR image (cv2)
        self.current_image = None       # Image after transformation
        self.display_scale = 1.0        # Fixed scale for display only

        # ---- Build GUI ----
        self.build_gui()

    # ------------------------------------------------------------
    # GUI Construction
    # ------------------------------------------------------------

    def build_gui(self):
        # Top bar
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        load_btn = tk.Button(top_frame, text="Load Image", command=self.load_image)
        load_btn.pack(side=tk.LEFT, padx=5, pady=5)

        reset_btn = tk.Button(top_frame, text="Reset Sliders", command=self.reset_sliders)
        reset_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Canvas (image display)
        self.canvas = tk.Canvas(self.root, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Sliders frame
        sliders = tk.Frame(self.root)
        sliders.pack(side=tk.BOTTOM, fill=tk.X)

        self.rot_slider = tk.Scale(
            sliders, from_=-180, to=180,
            orient=tk.HORIZONTAL,
            label="Rotation (degrees)",
            command=self.on_slider_change
        )
        self.rot_slider.pack(fill=tk.X)

        self.sx_slider = tk.Scale(
            sliders, from_=0.1, to=3.0,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            label="Scale X",
            command=self.on_slider_change
        )
        self.sx_slider.set(1.0)
        self.sx_slider.pack(fill=tk.X)

        self.sy_slider = tk.Scale(
            sliders, from_=0.1, to=3.0,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            label="Scale Y",
            command=self.on_slider_change
        )
        self.sy_slider.set(1.0)
        self.sy_slider.pack(fill=tk.X)

    # ------------------------------------------------------------
    # Image Loading
    # ------------------------------------------------------------

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")]
        )
        if not file_path:
            return

        img = cv2.imread(file_path)
        if img is None:
            return

        self.original_image = img
        self.current_image = img.copy()

        # Compute display scale ONCE (fit-to-window)
        self.root.update_idletasks()

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        h, w = img.shape[:2]

        scale_w = canvas_w / w
        scale_h = canvas_h / h
        self.display_scale = min(scale_w, scale_h, 1.0)

        self.show_on_canvas(self.current_image)

    # ------------------------------------------------------------
    # Slider Handling
    # ------------------------------------------------------------

    def reset_sliders(self):
        self.rot_slider.set(0)
        self.sx_slider.set(1.0)
        self.sy_slider.set(1.0)

    def on_slider_change(self, _=None):
        if self.original_image is None:
            return

        angle = self.rot_slider.get()
        sx = self.sx_slider.get()
        sy = self.sy_slider.get()

        # --------------------------------------------------------
        # TODO FOR STUDENTS:
        #
        # 1. Build rotation matrix R(theta)
        # 2. Build scaling matrix S(sx, sy)
        # 3. Compose A = R @ S  (or chosen order)
        # 4. Wrap with center translation:
        #       M = T(cx, cy) @ A @ T(-cx, -cy)
        # 5. Perform backward mapping
        # 6. Implement interpolation (nearest / bilinear)
        #
        # For now we simply copy the original image.
        # --------------------------------------------------------

        warp.warp_image(self.original_image, angle, sx, sy)
    
        transformed = self.original_image.copy()

        self.current_image = transformed
        self.show_on_canvas(self.current_image)

    # ------------------------------------------------------------
    # Display Logic
    # ------------------------------------------------------------

    def show_on_canvas(self, img_bgr):
        # Convert BGR (OpenCV) -> RGB (Pillow)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        h, w = img_rgb.shape[:2]

        # Apply fixed display scaling
        if self.display_scale != 1.0:
            img_rgb = cv2.resize(
                img_rgb,
                (int(w * self.display_scale), int(h * self.display_scale)),
                interpolation=cv2.INTER_AREA
            )

        # Convert NumPy -> PIL
        pil_img = Image.fromarray(img_rgb)

        # PIL -> Tk
        photo = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=photo)
        self.canvas.image = photo  # prevent garbage collection


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()