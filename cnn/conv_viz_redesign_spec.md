# Claude Code Prompt — Modify `draw_conv_viz.py`

Modify `draw_conv_viz.py` to redesign the visualization. Here is the full description of the new layout:

---

## Overall layout (left to right)

1. Input cube (8×14×14)
2. Kernel (8×3×3) floating in front of the input, aligned with the highlighted patch
3. Output feature map (1×14×14 slab) with one highlighted pixel

---

## Cell size

Use the same `CELL` size for both the input cube and the kernel. Remove `KERNEL_CELL` — there is only one cell size now.

---

## Kernel positioning

The kernel should appear to float just in front of the input cube, slightly detached. In isometric 3D space, "in front" means the kernel is at the same row/col position as the highlighted patch, but offset toward the viewer — i.e. shifted in screen space: lower and to the right by a small margin (e.g. one or two cell widths). The kernel should not overlap the input cube visually. Experiment with the offset to find a clean separation.

---

## Connecting lines (back corners of kernel → patch corners on input)

Draw lines connecting only the **two back corners** of the kernel's back face to the corresponding corners of the 3×3 patch on the input cube's top face. "Back corners" means the two corners of the kernel that face toward the input cube.

Use the **painter's algorithm** to simulate solid-outside / dashed-inside appearance, in this order:

1. Draw the lines as **solid** (thin, neutral color, e.g. gray)
2. Draw the input cube
3. Draw the kernel
4. Draw the **same lines again** as **dashed** (same coordinates, same color)

This way the cube faces paint over the solid lines where they overlap, and the dashed lines on top give the "hidden edge" convention inside the cube.

---

## Output feature map

To the right of the kernel, draw a single-channel slab: a flat cube of shape **1×14×14** (depth=1, rows=14, cols=14). Use the same `CELL` size. Highlight the single pixel at position **(PATCH_R, PATCH_C)** in the same amber color used for the patch in the input. Leave all other pixels in a neutral output color (e.g. light green or light gray). Add a label below: "Output channel" and "14 × 14".

---

## Labels

- Keep the existing labels for the input cube ("Input feature map", "8 × 14 × 14", "depth=8")
- Add labels for the kernel ("Kernel", "8 × 3 × 3", "depth=8")
- Add labels for the output slab ("Output channel", "14 × 14")
- Remove the dashed arrow that currently connects the patch to the kernel — the connecting lines replace it

---

## General notes

- The connecting line coordinates should be computed once and stored, then reused for both the solid and dashed draw passes
- Keep all existing helper functions (`iso`, `top_face`, `right_face`, `left_face`, `pts`, `build_cube`)
- The output file should remain `conv_input_kernel.svg` (or make it configurable via `sys.argv`)
