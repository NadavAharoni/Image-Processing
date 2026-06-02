# ─────────────────────────────────────────────────────────────────────────────
# draw_kernel.py
#
# Draws an isometric 3D box representing a conv kernel of shape D×H×W.
# Each unit cell is drawn as a small cube in isometric projection.
#
# Usage:  python draw_kernel.py            → kernel_8x3x3.svg
#         python draw_kernel.py out.svg    → out.svg
# ─────────────────────────────────────────────────────────────────────────────

import sys, math
from pathlib import Path

# ── KERNEL SHAPE ─────────────────────────────────────────────────────────────

DEPTH  = 8   # number of input channels  (D)
ROWS   = 3   # kernel height             (H)
COLS   = 3   # kernel width              (W)

# ── STYLE ─────────────────────────────────────────────────────────────────────

CELL       = 28          # size of one cube face (px)
LABEL_GAP  = 48          # space below box for labels

# Face colours (light, mid, dark for top / right / left faces)
TOP_FILL   = "#EEEDFE"
TOP_STROKE = "#7F77DD"
RIGHT_FILL = "#C8C4F5"
RIGHT_STROKE = "#7F77DD"
LEFT_FILL  = "#A8A3EC"
LEFT_STROKE  = "#7F77DD"
TEXT_COLOR = "#3C3489"

# ─────────────────────────────────────────────────────────────────────────────
# Isometric helpers
# ─────────────────────────────────────────────────────────────────────────────
# We use a simple isometric basis:
#   x-axis (cols)  → right and slightly down
#   y-axis (rows)  → left and slightly down
#   z-axis (depth) → straight up
#
# iso_x, iso_y: screen coords of the front-bottom-left corner of cell (d, r, c)

def iso(d, r, c, cell):
    """Screen (x,y) of the origin of cell at depth d, row r, col c."""
    sx = (c - r) * cell * math.cos(math.radians(30))
    sy = (c + r) * cell * math.sin(math.radians(30)) - d * cell
    return sx, sy


def top_face(ox, oy, cell):
    """Four corners of the top face (parallelogram)."""
    dx = cell * math.cos(math.radians(30))
    dy = cell * math.sin(math.radians(30))
    return [
        (ox,        oy),
        (ox + dx,   oy + dy),
        (ox,        oy + 2*dy),
        (ox - dx,   oy + dy),
    ]


def right_face(ox, oy, cell):
    """Four corners of the right face."""
    dx = cell * math.cos(math.radians(30))
    dy = cell * math.sin(math.radians(30))
    return [
        (ox + dx,   oy + dy),
        (ox + dx,   oy + dy + cell),
        (ox,        oy + 2*dy + cell),
        (ox,        oy + 2*dy),
    ]


def left_face(ox, oy, cell):
    """Four corners of the left face."""
    dx = cell * math.cos(math.radians(30))
    dy = cell * math.sin(math.radians(30))
    return [
        (ox,        oy + 2*dy),
        (ox,        oy + 2*dy + cell),
        (ox - dx,   oy + dy + cell),
        (ox - dx,   oy + dy),
    ]


def pts(corners):
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in corners)


# ─────────────────────────────────────────────────────────────────────────────
# Build SVG
# ─────────────────────────────────────────────────────────────────────────────

def render(depth, rows, cols, cell, label_gap, out_path):
    lines = []

    # Collect all polygons first so we can compute bounding box
    polys = []   # (draw_order, face_type, corners)

    # Draw back-to-front, bottom-to-top so faces occlude correctly
    for d in range(depth):
        for r in range(rows - 1, -1, -1):
            for c in range(cols - 1, -1, -1):
                ox, oy = iso(d, r, c, cell)
                # Only draw visible faces:
                # top face: always visible for top layer (d == depth-1)
                #           and top row (r == 0) of other layers
                show_top   = (d == depth - 1)
                show_right = (c == cols - 1)
                show_left  = (r == rows - 1)

                order = d * rows * cols + (rows - 1 - r) * cols + (cols - 1 - c)

                if show_top:
                    polys.append((order, 'top',   top_face(ox, oy, cell)))
                if show_right:
                    polys.append((order, 'right', right_face(ox, oy, cell)))
                if show_left:
                    polys.append((order, 'left',  left_face(ox, oy, cell)))

    # Bounding box
    all_x = [x for _, _, corners in polys for x, y in corners]
    all_y = [y for _, _, corners in polys for x, y in corners]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    pad    = 20
    left_pad = 80   # extra room for the depth label on the left side
    width  = max_x - min_x + pad + left_pad
    height = max_y - min_y + pad * 2 + label_gap

    # Offset so everything sits within the canvas
    ox_off = -min_x + left_pad
    oy_off = -min_y + pad

    lines.append(f'<svg width="{width:.0f}" height="{height:.0f}" '
                 f'viewBox="0 0 {width:.0f} {height:.0f}" '
                 f'xmlns="http://www.w3.org/2000/svg">')
    lines.append('  <title>Convolutional kernel tensor</title>')

    # Draw faces sorted by order
    for _, face_type, corners in sorted(polys, key=lambda x: x[0]):
        shifted = [(x + ox_off, y + oy_off) for x, y in corners]
        if face_type == 'top':
            fill, stroke = TOP_FILL, TOP_STROKE
        elif face_type == 'right':
            fill, stroke = RIGHT_FILL, RIGHT_STROKE
        else:
            fill, stroke = LEFT_FILL, LEFT_STROKE
        lines.append(f'  <polygon points="{pts(shifted)}" '
                     f'fill="{fill}" stroke="{stroke}" stroke-width="0.8"/>')

    # ── Labels ────────────────────────────────────────────────────────────────
    # Centre of the box bottom edge
    cx = (min_x + max_x) / 2 + ox_off
    ly = max_y + oy_off + 20

    lines.append(f'  <text x="{cx:.1f}" y="{ly:.1f}" '
                 f'text-anchor="middle" '
                 f'font-family="system-ui,sans-serif" font-size="15" '
                 f'font-weight="600" fill="{TEXT_COLOR}">'
                 f'{depth}×{rows}×{cols} kernel</text>')

    dx_ann = cell * math.cos(math.radians(30))
    dy_ann = cell * math.sin(math.radians(30))

    def ann(x, y, text, anchor="middle"):
        lines.append(f'  <text x="{x + ox_off:.1f}" y="{y + oy_off:.1f}" '
                     f'text-anchor="{anchor}" dominant-baseline="central" '
                     f'font-family="system-ui,sans-serif" font-size="11" '
                     f'fill="{TEXT_COLOR}" opacity="0.75">{text}</text>')

    # depth label — left side, middle channel
    d_mid = depth / 2
    lx, ly2 = iso(d_mid, rows - 1, 0, cell)
    ann(lx - dx_ann - 6, ly2 + dy_ann + cell/2, f"depth={depth}", "end")

    # cols label — left face (r=rows-1) spans all cols; label its bottom edge at d=0
    # The left face at (d=0, r=rows-1) runs from c=0 (right bottom) to c=cols-1 (left bottom).
    # bottom-right corner of left face at c=0:      (lf_rx0, lf_ry0+2*dy_ann+cell)
    # bottom-left  corner of left face at c=cols-1: (lf_lx1-dx_ann, lf_ly1+dy_ann+cell)
    lf_rx0, lf_ry0 = iso(0, rows-1, 0, cell)           # right end  (c=0)
    lf_lx1, lf_ly1 = iso(0, rows-1, cols-1, cell)      # left  end  (c=cols-1)
    col_mid_x = (lf_rx0 + lf_lx1 - dx_ann) / 2
    col_mid_y = (lf_ry0 + 2*dy_ann + cell + lf_ly1 + dy_ann + cell) / 2
    ann(col_mid_x - 4, col_mid_y + 10, f"cols={cols}", "end")

    # rows label — right face (c=cols-1) spans all rows; label its bottom edge at d=0
    # The right face at (d=0, c=cols-1) runs from r=0 (top-right) to r=rows-1 (bottom-left).
    # bottom-right corner of right face at r=0:      (rf_rx0+dx_ann, rf_ry0+dy_ann+cell)
    # bottom-left  corner of right face at r=rows-1: (rf_lx1, rf_ly1+2*dy_ann+cell)
    rf_rx0, rf_ry0 = iso(0, 0, cols-1, cell)           # top    end  (r=0)
    rf_lx1, rf_ly1 = iso(0, rows-1, cols-1, cell)      # bottom end  (r=rows-1)
    row_mid_x = (rf_rx0 + dx_ann + rf_lx1) / 2
    row_mid_y = (rf_ry0 + dy_ann + cell + rf_ly1 + 2*dy_ann + cell) / 2
    ann(row_mid_x + 4, row_mid_y + 10, f"rows={rows}", "start")

    lines.append('</svg>')
    Path(out_path).write_text('\n'.join(lines), encoding='utf-8')
    print(f"SVG written to {out_path}")


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else f"kernel_{DEPTH}x{ROWS}x{COLS}.svg"
    render(DEPTH, ROWS, COLS, CELL, LABEL_GAP, out)
