import math
from pathlib import Path

CELL = 18          # smaller cell for the large input cube
KERNEL_CELL = 28   # larger for the kernel (matches original)

# Colors - input cube
IN_TOP_FILL    = "#D4EAF7"
IN_TOP_STROKE  = "#4A90C4"
IN_RIGHT_FILL  = "#A8D1EE"
IN_RIGHT_STROKE= "#4A90C4"
IN_LEFT_FILL   = "#7CBDE5"
IN_LEFT_STROKE = "#4A90C4"

# Colors - highlighted patch inside input
HL_TOP_FILL    = "#FFD580"
HL_TOP_STROKE  = "#B07D00"
HL_RIGHT_FILL  = "#FFBC40"
HL_RIGHT_STROKE= "#B07D00"
HL_LEFT_FILL   = "#E8A200"
HL_LEFT_STROKE = "#B07D00"

# Colors - kernel
KN_TOP_FILL    = "#EEEDFE"
KN_TOP_STROKE  = "#7F77DD"
KN_RIGHT_FILL  = "#C8C4F5"
KN_RIGHT_STROKE= "#7F77DD"
KN_LEFT_FILL   = "#A8A3EC"
KN_LEFT_STROKE = "#7F77DD"

TEXT_COLOR     = "#1A3A5C"
ARROW_COLOR    = "#888"

def iso(d, r, c, cell):
    sx = (c - r) * cell * math.cos(math.radians(30))
    sy = (c + r) * cell * math.sin(math.radians(30)) - d * cell
    return sx, sy

def top_face(ox, oy, cell):
    dx = cell * math.cos(math.radians(30))
    dy = cell * math.sin(math.radians(30))
    return [(ox, oy), (ox+dx, oy+dy), (ox, oy+2*dy), (ox-dx, oy+dy)]

def right_face(ox, oy, cell):
    dx = cell * math.cos(math.radians(30))
    dy = cell * math.sin(math.radians(30))
    return [(ox+dx, oy+dy), (ox+dx, oy+dy+cell), (ox, oy+2*dy+cell), (ox, oy+2*dy)]

def left_face(ox, oy, cell):
    dx = cell * math.cos(math.radians(30))
    dy = cell * math.sin(math.radians(30))
    return [(ox, oy+2*dy), (ox, oy+2*dy+cell), (ox-dx, oy+dy+cell), (ox-dx, oy+dy)]

def pts(corners):
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in corners)

def build_cube(depth, rows, cols, cell, highlight_patch=None):
    """
    Build list of (order, face_type, corners, is_highlight) for the cube.
    highlight_patch = (r0, c0, h, w) - top-left corner and size of highlighted region
    """
    polys = []
    for d in range(depth):
        for r in range(rows - 1, -1, -1):
            for c in range(cols - 1, -1, -1):
                ox, oy = iso(d, r, c, cell)
                show_top   = (d == depth - 1)
                show_right = (c == cols - 1)
                show_left  = (r == rows - 1)
                order = d * rows * cols + (rows - 1 - r) * cols + (cols - 1 - c)

                # Check if this cell is in the highlight patch
                is_hl = False
                if highlight_patch:
                    r0, c0, ph, pw = highlight_patch
                    is_hl = (r0 <= r < r0+ph) and (c0 <= c < c0+pw)

                if show_top:
                    polys.append((order, 'top', top_face(ox, oy, cell), is_hl))
                if show_right:
                    polys.append((order, 'right', right_face(ox, oy, cell), is_hl))
                if show_left:
                    polys.append((order, 'left', left_face(ox, oy, cell), is_hl))
    return polys

def bbox(polys):
    all_x = [x for _, _, corners, _ in polys for x, _ in corners]
    all_y = [y for _, _, corners, _ in polys for _, y in corners]
    return min(all_x), max(all_x), min(all_y), max(all_y)

def face_colors(face_type, is_hl, is_input):
    if is_input:
        if is_hl:
            fills = {'top': HL_TOP_FILL, 'right': HL_RIGHT_FILL, 'left': HL_LEFT_FILL}
            strokes = {'top': HL_TOP_STROKE, 'right': HL_RIGHT_STROKE, 'left': HL_LEFT_STROKE}
        else:
            fills = {'top': IN_TOP_FILL, 'right': IN_RIGHT_FILL, 'left': IN_LEFT_FILL}
            strokes = {'top': IN_TOP_STROKE, 'right': IN_RIGHT_STROKE, 'left': IN_LEFT_STROKE}
    else:
        fills = {'top': KN_TOP_FILL, 'right': KN_RIGHT_FILL, 'left': KN_LEFT_FILL}
        strokes = {'top': KN_TOP_STROKE, 'right': KN_RIGHT_STROKE, 'left': KN_LEFT_STROKE}
    return fills[face_type], strokes[face_type]


def main():
    lines = []

    # ── Build input cube (8×14×14) ───────────────────────────────────────────────
    IN_D, IN_R, IN_C = 8, 14, 14
    # Highlight patch starting at (row=2, col=6), size 3×3
    PATCH_R, PATCH_C = 2, 6

    in_polys = build_cube(IN_D, IN_R, IN_C, CELL, highlight_patch=(PATCH_R, PATCH_C, 3, 3))

    # ── Build kernel cube (8×3×3) ────────────────────────────────────────────────
    KN_D, KN_R, KN_C = 8, 3, 3
    kn_polys = build_cube(KN_D, KN_R, KN_C, KERNEL_CELL)

    # ── Compute bounding boxes ────────────────────────────────────────────────────
    in_minx, in_maxx, in_miny, in_maxy = bbox(in_polys)
    kn_minx, kn_maxx, kn_miny, kn_maxy = bbox(kn_polys)

    PAD = 30
    LABEL_GAP = 60
    GAP_BETWEEN = 80  # horizontal gap between the two cubes

    in_w = in_maxx - in_minx
    kn_w = kn_maxx - kn_minx
    in_h = in_maxy - in_miny
    kn_h = kn_maxy - kn_miny

    # Left margin for depth label
    LEFT_MARGIN = 80

    # Input cube offset
    in_ox = LEFT_MARGIN - in_minx
    in_oy = PAD - in_miny

    # Kernel cube offset (placed to the right of input)
    kn_ox = in_ox + in_w + GAP_BETWEEN - kn_minx
    kn_oy = PAD + (in_h - kn_h) / 2 - kn_miny  # vertically center kernel to input

    total_w = kn_ox + kn_w + kn_minx + PAD + 40
    total_h = max(in_h, kn_h) + PAD * 2 + LABEL_GAP

    lines.append(f'<svg width="{total_w:.0f}" height="{total_h:.0f}" '
                 f'viewBox="0 0 {total_w:.0f} {total_h:.0f}" '
                 f'xmlns="http://www.w3.org/2000/svg">')
    lines.append('  <title>Convolution: input and kernel</title>')

    def draw_polys(polys, ox, oy, is_input, sw=0.8):
        for _, face_type, corners, is_hl in sorted(polys, key=lambda x: x[0]):
            shifted = [(x + ox, y + oy) for x, y in corners]
            fill, stroke = face_colors(face_type, is_hl, is_input)
            lines.append(f'  <polygon points="{pts(shifted)}" '
                         f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    draw_polys(in_polys, in_ox, in_oy, is_input=True, sw=0.6)
    draw_polys(kn_polys, kn_ox, kn_oy, is_input=False, sw=0.9)

    # ── Draw connection arrow between highlighted patch and kernel ────────────────
    dx_ann = CELL * math.cos(math.radians(30))
    dy_ann = CELL * math.sin(math.radians(30))

    # Right edge of highlighted patch (top layer, col = PATCH_C+3-1 = right col, center row of patch)
    patch_right_r = PATCH_R + 1  # middle row of patch
    patch_right_c = PATCH_C + 3 - 1  # rightmost col
    ox_pr, oy_pr = iso(IN_D - 1, patch_right_r, patch_right_c, CELL)
    # Right corner of the top face
    arr_start_x = ox_pr + dx_ann + in_ox
    arr_start_y = oy_pr + dy_ann + in_oy

    # Left edge of kernel (top layer)
    kn_left_r, kn_left_c = KN_R - 1, 0
    ox_kl, oy_kl = iso(KN_D - 1, kn_left_r, kn_left_c, KERNEL_CELL)
    arr_end_x = ox_kl - KERNEL_CELL * math.cos(math.radians(30)) + kn_ox
    arr_end_y = oy_kl + KERNEL_CELL * math.sin(math.radians(30)) + kn_oy

    # Draw a curved arrow
    mid_x = (arr_start_x + arr_end_x) / 2
    mid_y = min(arr_start_y, arr_end_y) - 18

    lines.append(f'  <defs>'
                 f'<marker id="arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">'
                 f'<polygon points="0 0, 8 3, 0 6" fill="{ARROW_COLOR}"/>'
                 f'</marker></defs>')
    lines.append(f'  <path d="M {arr_start_x:.1f},{arr_start_y:.1f} '
                 f'Q {mid_x:.1f},{mid_y:.1f} {arr_end_x:.1f},{arr_end_y:.1f}" '
                 f'fill="none" stroke="{ARROW_COLOR}" stroke-width="1.5" stroke-dasharray="5,3" '
                 f'marker-end="url(#arrowhead)"/>')

    # ── Labels ────────────────────────────────────────────────────────────────────
    def text(x, y, s, anchor="middle", size=13, weight="600", color=TEXT_COLOR, opacity=1.0):
        lines.append(f'  <text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
                     f'font-family="system-ui,sans-serif" font-size="{size}" '
                     f'font-weight="{weight}" fill="{color}" opacity="{opacity}">{s}</text>')

    # Input cube label
    in_cx = in_ox + (in_minx + in_maxx) / 2
    text(in_cx, PAD + in_h + in_oy - in_miny + 22, "Input feature map", size=14)
    text(in_cx, PAD + in_h + in_oy - in_miny + 40, "8 × 14 × 14", size=12, weight="400", opacity=0.75)

    # Kernel label
    kn_cx = kn_ox + (kn_minx + kn_maxx) / 2
    text(kn_cx, PAD + kn_h + kn_oy - kn_miny + 22, "Kernel (filter)", size=14)
    text(kn_cx, PAD + kn_h + kn_oy - kn_miny + 40, "8 × 3 × 3", size=12, weight="400", opacity=0.75)

    # Depth annotations (left side of each cube)
    def depth_ann(r, cell, ox, oy, label, depth):
        d_mid = depth / 2
        lx, ly = iso(d_mid, r, 0, cell)
        dx = cell * math.cos(math.radians(30))
        dy = cell * math.sin(math.radians(30))
        text(lx - dx - 8 + ox, ly + dy + cell/2 + oy, label, anchor="end", size=11, weight="400", opacity=0.75)

    depth_ann(IN_R-1, CELL, in_ox, in_oy, f"depth={IN_D}", IN_D)
    depth_ann(KN_R-1, KERNEL_CELL, kn_ox, kn_oy, f"depth={KN_D}", KN_D)

    # Patch annotation
    patch_top_r, patch_top_c = PATCH_R, PATCH_C
    ox_pt, oy_pt = iso(IN_D - 1, patch_top_r, patch_top_c, CELL)
    # top-left of highlighted patch on top layer
    hl_label_x = ox_pt - CELL * math.cos(math.radians(30)) + in_ox
    hl_label_y = oy_pt + CELL * math.sin(math.radians(30)) + in_oy - 14
    text(hl_label_x - 6, hl_label_y - 6, "3×3 patch", anchor="end", size=11,
         weight="400", color="#8B6000", opacity=0.9)

    lines.append('</svg>')

    out = Path(__file__).parent / "conv_input_kernel.svg"
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(f"Written: {out}")


if __name__ == "__main__":
    main()
