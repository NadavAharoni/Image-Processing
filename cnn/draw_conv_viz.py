import math
import sys
from pathlib import Path

CELL = 18

# Colors - input cube
IN_TOP_FILL     = "#D4EAF7"; IN_TOP_STROKE     = "#4A90C4"
IN_RIGHT_FILL   = "#A8D1EE"; IN_RIGHT_STROKE   = "#4A90C4"
IN_LEFT_FILL    = "#7CBDE5"; IN_LEFT_STROKE    = "#4A90C4"

# Colors - highlighted patch (input) and highlighted output pixel
HL_TOP_FILL     = "#FFD580"; HL_TOP_STROKE     = "#B07D00"
HL_RIGHT_FILL   = "#FFBC40"; HL_RIGHT_STROKE   = "#B07D00"
HL_LEFT_FILL    = "#E8A200"; HL_LEFT_STROKE    = "#B07D00"

# Colors - kernel
KN_TOP_FILL     = "#EEEDFE"; KN_TOP_STROKE     = "#7F77DD"
KN_RIGHT_FILL   = "#C8C4F5"; KN_RIGHT_STROKE   = "#7F77DD"
KN_LEFT_FILL    = "#A8A3EC"; KN_LEFT_STROKE    = "#7F77DD"

# Colors - output feature map
OUT_TOP_FILL    = "#E0F5E0"; OUT_TOP_STROKE    = "#5A9A5A"
OUT_RIGHT_FILL  = "#C0E8C0"; OUT_RIGHT_STROKE  = "#5A9A5A"
OUT_LEFT_FILL   = "#A0DBB0"; OUT_LEFT_STROKE   = "#5A9A5A"

TEXT_COLOR  = "#1A3A5C"
CONN_COLOR  = "#999999"


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
    polys = []
    for d in range(depth):
        for r in range(rows - 1, -1, -1):
            for c in range(cols - 1, -1, -1):
                ox, oy = iso(d, r, c, cell)
                show_top   = (d == depth - 1)
                show_right = (c == cols - 1)
                show_left  = (r == rows - 1)
                order = d * rows * cols + (rows - 1 - r) * cols + (cols - 1 - c)

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

def face_colors(face_type, is_hl, cube_type):
    if cube_type == 'input':
        if is_hl:
            fills   = {'top': HL_TOP_FILL,   'right': HL_RIGHT_FILL,   'left': HL_LEFT_FILL}
            strokes = {'top': HL_TOP_STROKE,  'right': HL_RIGHT_STROKE, 'left': HL_LEFT_STROKE}
        else:
            fills   = {'top': IN_TOP_FILL,   'right': IN_RIGHT_FILL,   'left': IN_LEFT_FILL}
            strokes = {'top': IN_TOP_STROKE,  'right': IN_RIGHT_STROKE, 'left': IN_LEFT_STROKE}
    elif cube_type == 'kernel':
        fills   = {'top': KN_TOP_FILL,   'right': KN_RIGHT_FILL,   'left': KN_LEFT_FILL}
        strokes = {'top': KN_TOP_STROKE,  'right': KN_RIGHT_STROKE, 'left': KN_LEFT_STROKE}
    elif cube_type == 'output':
        if is_hl:
            fills   = {'top': HL_TOP_FILL,   'right': HL_RIGHT_FILL,   'left': HL_LEFT_FILL}
            strokes = {'top': HL_TOP_STROKE,  'right': HL_RIGHT_STROKE, 'left': HL_LEFT_STROKE}
        else:
            fills   = {'top': OUT_TOP_FILL,   'right': OUT_RIGHT_FILL,   'left': OUT_LEFT_FILL}
            strokes = {'top': OUT_TOP_STROKE,  'right': OUT_RIGHT_STROKE, 'left': OUT_LEFT_STROKE}
    return fills[face_type], strokes[face_type]


def main():
    out_file = sys.argv[1] if len(sys.argv) > 1 else "conv_input_kernel.svg"
    lines = []

    IN_D,  IN_R,  IN_C  = 8, 14, 14
    KN_D,  KN_R,  KN_C  = 8,  3,  3
    OUT_D, OUT_R, OUT_C  = 1, 14, 14
    PATCH_R, PATCH_C = 2, 6

    dx = CELL * math.cos(math.radians(30))
    dy = CELL * math.sin(math.radians(30))

    in_polys  = build_cube(IN_D,  IN_R,  IN_C,  CELL, highlight_patch=(PATCH_R, PATCH_C, 3, 3))
    kn_polys  = build_cube(KN_D,  KN_R,  KN_C,  CELL)
    out_polys = build_cube(OUT_D, OUT_R, OUT_C, CELL, highlight_patch=(PATCH_R, PATCH_C, 1, 1))

    in_minx,  in_maxx,  in_miny,  in_maxy  = bbox(in_polys)
    kn_minx,  kn_maxx,  kn_miny,  kn_maxy  = bbox(kn_polys)
    out_minx, out_maxx, out_miny, out_maxy = bbox(out_polys)

    PAD           = 30
    GAP_KN_INPUT  = 80   # horizontal gap between kernel right edge and input left edge
    GAP_OUTPUT    = 70   # horizontal gap between input right edge and output left edge
    LABEL_GAP     = 65

    # Kernel: left edge at x=PAD
    kn_ox = PAD - kn_minx

    # Input: placed to the right of the kernel
    in_ox = kn_ox + kn_maxx + GAP_KN_INPUT - in_minx
    in_oy = PAD - in_miny

    # Kernel: vertically centered to input
    in_screen_cy = in_oy + (in_miny + in_maxy) / 2
    kn_oy = in_screen_cy - (kn_miny + kn_maxy) / 2

    # Output cube: right of the input right edge
    out_ox = in_ox + in_maxx + GAP_OUTPUT - out_minx
    out_oy = in_screen_cy - (out_miny + out_maxy) / 2

    total_w = out_ox + out_maxx + PAD + 20
    total_h = max(in_oy + in_maxy, kn_oy + kn_maxy, out_oy + out_maxy) + LABEL_GAP + PAD

    lines.append(f'<svg width="{total_w:.0f}" height="{total_h:.0f}" '
                 f'viewBox="0 0 {total_w:.0f} {total_h:.0f}" '
                 f'xmlns="http://www.w3.org/2000/svg">')
    lines.append('  <title>Convolution: input, kernel, and output</title>')

    def draw_polys(polys, ox, oy, cube_type, sw=0.8):
        for _, face_type, corners, is_hl in sorted(polys, key=lambda x: x[0]):
            shifted = [(x + ox, y + oy) for x, y in corners]
            fill, stroke = face_colors(face_type, is_hl, cube_type)
            lines.append(f'  <polygon points="{pts(shifted)}" '
                         f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    # ── Connecting line endpoints (computed once, reused for both draw passes) ───
    # Kernel back corners: the two corners of the r=0 edge of the kernel top face
    # (the face pointing toward the input, which is to the right).
    # Corner 1: far/top vertex of kernel top-face rhombus (r=0, c=0)
    kc1x, kc1y = iso(KN_D-1, 0, 0, CELL)
    kc1 = (kc1x + kn_ox, kc1y + kn_oy)

    # Corner 2: right vertex of top-face rhombus (r=0, c=KN_C-1, right extension)
    kc2ox, kc2oy = iso(KN_D-1, 0, KN_C-1, CELL)
    kc2 = (kc2ox + dx + kn_ox, kc2oy + dy + kn_oy)

    # Corresponding patch corners on input top face (same r=PATCH_R, corresponding cols)
    pc1x, pc1y = iso(IN_D-1, PATCH_R, PATCH_C, CELL)
    pc1 = (pc1x + in_ox, pc1y + in_oy)

    pc2ox, pc2oy = iso(IN_D-1, PATCH_R, PATCH_C + KN_C - 1, CELL)
    pc2 = (pc2ox + dx + in_ox, pc2oy + dy + in_oy)

    conn_lines = [(kc1, pc1), (kc2, pc2)]

    def draw_conn_lines(dashed=False):
        dash = ' stroke-dasharray="4,3"' if dashed else ''
        for (x1, y1), (x2, y2) in conn_lines:
            lines.append(f'  <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                         f'stroke="{CONN_COLOR}" stroke-width="1.2"{dash}/>')

    # ── Painter's algorithm ───────────────────────────────────────────────────────
    draw_conn_lines(dashed=False)   # 1. solid lines (behind everything)
    draw_polys(in_polys, in_ox, in_oy, 'input', sw=0.6)   # 2. input cube
    draw_polys(kn_polys, kn_ox, kn_oy, 'kernel', sw=0.9)  # 3. kernel
    draw_conn_lines(dashed=True)    # 4. dashed lines (hidden-edge convention)
    draw_polys(out_polys, out_ox, out_oy, 'output', sw=0.6)  # 5. output slab

    # ── Labels ────────────────────────────────────────────────────────────────────
    def text(x, y, s, anchor="middle", size=13, weight="600", color=TEXT_COLOR, opacity=1.0):
        lines.append(f'  <text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
                     f'font-family="system-ui,sans-serif" font-size="{size}" '
                     f'font-weight="{weight}" fill="{color}" opacity="{opacity}">{s}</text>')

    in_cx = in_ox + (in_minx + in_maxx) / 2
    in_label_y = in_oy + in_maxy + 22
    text(in_cx, in_label_y,      "Input feature map", size=14)
    text(in_cx, in_label_y + 18, "8 × 14 × 14", size=12, weight="400", opacity=0.75)

    kn_cx = kn_ox + (kn_minx + kn_maxx) / 2
    kn_label_y = kn_oy + kn_maxy + 22
    text(kn_cx, kn_label_y,      "Kernel", size=14)
    text(kn_cx, kn_label_y + 18, "8 × 3 × 3", size=12, weight="400", opacity=0.75)

    out_cx = out_ox + (out_minx + out_maxx) / 2
    out_label_y = out_oy + out_maxy + 22
    text(out_cx, out_label_y,      "Output channel", size=14)
    text(out_cx, out_label_y + 18, "14 × 14", size=12, weight="400", opacity=0.75)

    # Depth annotations (left side of input and kernel)
    def depth_ann(rows, depth, cell, ox, oy, label):
        d_mid = depth / 2
        lx, ly = iso(d_mid, rows-1, 0, cell)
        text(lx - dx - 8 + ox, ly + dy + cell/2 + oy, label,
             anchor="end", size=11, weight="400", opacity=0.75)

    depth_ann(IN_R, IN_D, CELL, in_ox, in_oy, f"depth={IN_D}")
    depth_ann(KN_R, KN_D, CELL, kn_ox, kn_oy, f"depth={KN_D}")

    # Patch annotation
    pox, poy = iso(IN_D-1, PATCH_R, PATCH_C, CELL)
    text(pox - dx - 6 + in_ox, poy + dy + in_oy - 20, "3×3 patch",
         anchor="end", size=11, weight="400", color="#8B6000", opacity=0.9)

    lines.append('</svg>')

    out = Path(__file__).parent / out_file
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(f"Written: {out}")


if __name__ == "__main__":
    main()
