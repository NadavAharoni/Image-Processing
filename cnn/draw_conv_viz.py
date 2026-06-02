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
    KERN_LEFT     = 90   # extra left room for the kernel depth annotation text
    GAP_KN_INPUT  = 80   # horizontal gap between kernel right edge and input left edge
    GAP_OUTPUT    = 70   # horizontal gap between input right edge and output left edge
    LABEL_GAP     = 65

    # Kernel: left edge at x=KERN_LEFT (enough room for the depth label)
    kn_ox = KERN_LEFT - kn_minx

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
    # Line 1: kernel corner a → patch corner A  (far corners of both top faces)
    kc_a = (iso(KN_D-1, 0,        0,        CELL)[0] + kn_ox,
            iso(KN_D-1, 0,        0,        CELL)[1] + kn_oy)
    pc_A = (iso(IN_D-1, PATCH_R,  PATCH_C,  CELL)[0] + in_ox,
            iso(IN_D-1, PATCH_R,  PATCH_C,  CELL)[1] + in_oy)

    # Line 2: kernel corner c → patch corner C  (near corners of both top faces)
    kc_c = (iso(KN_D-1, KN_R-1,           KN_C-1,           CELL)[0] + kn_ox,
            iso(KN_D-1, KN_R-1,           KN_C-1,           CELL)[1] + 2*dy + kn_oy)
    pc_C = (iso(IN_D-1, PATCH_R+KN_R-1,   PATCH_C+KN_C-1,   CELL)[0] + in_ox,
            iso(IN_D-1, PATCH_R+KN_R-1,   PATCH_C+KN_C-1,   CELL)[1] + 2*dy + in_oy)

    # Line 3: kernel c corner at d=0 (bottom of kernel) → same position at d=0 in input (hidden)
    kc_cb = (iso(0, KN_R-1,           KN_C-1,           CELL)[0] + kn_ox,
             iso(0, KN_R-1,           KN_C-1,           CELL)[1] + 2*dy + CELL + kn_oy)
    pc_Cb = (iso(0, PATCH_R+KN_R-1,   PATCH_C+KN_C-1,   CELL)[0] + in_ox,
             iso(0, PATCH_R+KN_R-1,   PATCH_C+KN_C-1,   CELL)[1] + 2*dy + CELL + in_oy)

    conn_lines = [(kc_a, pc_A), (kc_c, pc_C), (kc_cb, pc_Cb)]

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

    # ── Temporary corner labels ───────────────────────────────────────────────
    # Patch corners on input top face: A=far, B=right, C=near, D=left
    # Kernel top-face corners:         a=far, b=right, c=near, d=left
    def corner_dot(cx, cy, label, color, anchor="middle", dy_label=-8):
        r = 4
        lines.append(f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" '
                     f'fill="{color}" stroke="white" stroke-width="1"/>')
        lines.append(f'  <text x="{cx:.1f}" y="{cy+dy_label:.1f}" text-anchor="{anchor}" '
                     f'font-family="system-ui,sans-serif" font-size="11" '
                     f'font-weight="700" fill="{color}">{label}</text>')

    d = IN_D - 1
    # Patch: A=far, B=right, C=near, D=left
    Ax, Ay = iso(d, PATCH_R,        PATCH_C,            CELL)
    Bx, By = iso(d, PATCH_R,        PATCH_C+KN_C-1,     CELL); Bx += dx; By += dy
    Cx, Cy = iso(d, PATCH_R+KN_R-1, PATCH_C+KN_C-1,     CELL); Cy += 2*dy
    Dx, Dy = iso(d, PATCH_R+KN_R-1, PATCH_C,            CELL); Dx -= dx; Dy += dy

    PATCH_DOT = "#8B6000"
    corner_dot(Ax+in_ox, Ay+in_oy, "A", PATCH_DOT, dy_label=-10)
    corner_dot(Bx+in_ox, By+in_oy, "B", PATCH_DOT, anchor="start", dy_label=-4)
    corner_dot(Cx+in_ox, Cy+in_oy, "C", PATCH_DOT, dy_label=16)
    corner_dot(Dx+in_ox, Dy+in_oy, "D", PATCH_DOT, anchor="end", dy_label=-4)

    # Kernel top-face: a=far, b=right, c=near, d=left
    kd = KN_D - 1
    ax, ay = iso(kd, 0,        0,        CELL)
    bx, by = iso(kd, 0,        KN_C-1,   CELL); bx += dx; by += dy
    cx_, cy_ = iso(kd, KN_R-1, KN_C-1,   CELL); cy_ += 2*dy
    ddx, ddy = iso(kd, KN_R-1, 0,        CELL); ddx -= dx; ddy += dy

    KN_DOT = "#5A22CC"
    corner_dot(ax+kn_ox, ay+kn_oy, "a", KN_DOT, dy_label=-10)
    corner_dot(bx+kn_ox, by+kn_oy, "b", KN_DOT, anchor="start", dy_label=-4)
    corner_dot(cx_+kn_ox, cy_+kn_oy, "c", KN_DOT, dy_label=16)
    corner_dot(ddx+kn_ox, ddy+kn_oy, "d", KN_DOT, anchor="end", dy_label=-4)

    lines.append('</svg>')

    out = Path(__file__).parent / out_file
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(f"Written: {out}")


if __name__ == "__main__":
    main()
