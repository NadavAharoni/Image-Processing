# ─────────────────────────────────────────────────────────────────────────────
# nn_diagram.py
#
# Define a neural network as a list of layers, run the script, get an SVG.
# Usage:  python nn_diagram.py             → writes diagram.svg
#         python nn_diagram.py out.svg     → writes out.svg
# ─────────────────────────────────────────────────────────────────────────────

import sys
import math
from pathlib import Path

# ── NETWORK DEFINITION ────────────────────────────────────────────────────────
# Each layer is a dict with:
#   id       : unique string (used for edge references)
#   label    : bold text shown in the box
#   sublabel : smaller text shown below label (use "" for none)
#   type     : one of the keys in COLORS below

NETWORK = [
    {"id": "input",   "label": "Input",    "sublabel": "3×32×32",          "output": "",              "type": "input"},
    {"id": "conv1a",  "label": "Conv 1a",  "sublabel": "32 filters, 3×3",  "output": "→ 32×32×32",   "type": "conv"},
    {"id": "conv1b",  "label": "Conv 1b",  "sublabel": "32 filters, 3×3",  "output": "→ 32×32×32",   "type": "conv"},
    {"id": "pool1",   "label": "MaxPool",  "sublabel": "",                  "output": "→ 32×16×16",   "type": "pool"},
    {"id": "conv2a",  "label": "Conv 2a",  "sublabel": "64 filters, 3×3",  "output": "→ 64×16×16",   "type": "conv"},
    {"id": "conv2b",  "label": "Conv 2b",  "sublabel": "64 filters, 3×3",  "output": "→ 64×16×16",   "type": "conv"},
    {"id": "pool2",   "label": "MaxPool",  "sublabel": "",                  "output": "→ 64×8×8",     "type": "pool"},
    {"id": "conv3a",  "label": "Conv 3a",  "sublabel": "128 filters, 3×3", "output": "→ 128×8×8",    "type": "conv"},
    {"id": "conv3b",  "label": "Conv 3b",  "sublabel": "128 filters, 3×3", "output": "→ 128×8×8",    "type": "conv"},
    {"id": "pool3",   "label": "MaxPool",  "sublabel": "",                  "output": "→ 128×4×4",    "type": "pool"},
    {"id": "flatten", "label": "Flatten",  "sublabel": "",                  "output": "2048 units",    "type": "flatten"},
    {"id": "fc1",     "label": "FC layer", "sublabel": "2048 → 256",        "output": "ReLU → 256",    "type": "fc"},
    {"id": "fc2",     "label": "FC layer", "sublabel": "256 → 10",          "output": "",              "type": "fc"},
    {"id": "softmax", "label": "Softmax",  "sublabel": "10 classes",        "output": "",              "type": "output"},
]

# Edges: list of (from_id, to_id). Sequential by default — override here.
# Leave as None to auto-generate a simple chain from the NETWORK list order.
EDGES = None

# ── LAYOUT ────────────────────────────────────────────────────────────────────

NODES_PER_ROW = 4      # how many nodes before wrapping to next row
BOX_W         = 130    # box width  (px)
BOX_H         = 76     # box height (px) — fits title + sublabel + output
GAP_X         = 24     # horizontal gap between boxes
GAP_ROW       = 60     # vertical gap between rows

# ── COLORS (fill, stroke, title-text, subtitle-text) ─────────────────────────
# Light-mode hex values. Dark-mode variants handled via <style> block.

COLORS = {
    "input":   {"fill": "#F1EFE8", "stroke": "#888780", "title": "#444441", "sub": "#888780"},
    "conv":    {"fill": "#EEEDFE", "stroke": "#7F77DD", "title": "#3C3489", "sub": "#534AB7"},
    "pool":    {"fill": "#E1F5EE", "stroke": "#1D9E75", "title": "#085041", "sub": "#0F6E56"},
    "flatten": {"fill": "#F1EFE8", "stroke": "#888780", "title": "#444441", "sub": "#888780"},
    "fc":      {"fill": "#FAECE7", "stroke": "#D85A30", "title": "#712B13", "sub": "#993C1D"},
    "output":  {"fill": "#FAEEDA", "stroke": "#BA7517", "title": "#633806", "sub": "#854F0B"},
}

DARK_COLORS = {
    "input":   {"fill": "#2C2C2A", "stroke": "#888780", "title": "#D3D1C7", "sub": "#888780"},
    "conv":    {"fill": "#26215C", "stroke": "#AFA9EC", "title": "#CECBF6", "sub": "#AFA9EC"},
    "pool":    {"fill": "#04342C", "stroke": "#5DCAA5", "title": "#9FE1CB", "sub": "#5DCAA5"},
    "flatten": {"fill": "#2C2C2A", "stroke": "#888780", "title": "#D3D1C7", "sub": "#888780"},
    "fc":      {"fill": "#4A1B0C", "stroke": "#F0997B", "title": "#F5C4B3", "sub": "#D85A30"},
    "output":  {"fill": "#412402", "stroke": "#EF9F27", "title": "#FAC775", "sub": "#BA7517"},
}

# ── LEGEND ────────────────────────────────────────────────────────────────────

LEGEND_ITEMS = [
    ("conv",    "Convolutional layer"),
    ("pool",    "Pooling layer"),
    ("fc",      "Fully connected (FC)"),
    ("output",  "Output / softmax"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Layout engine
# ─────────────────────────────────────────────────────────────────────────────

def layout(network, nodes_per_row, box_w, box_h, gap_x, gap_row):
    """Assign (x, y, row, col_in_row) to each node. Snake direction per row."""
    positioned = []
    n = len(network)
    num_rows = math.ceil(n / nodes_per_row)

    for i, layer in enumerate(network):
        row = i // nodes_per_row
        col = i %  nodes_per_row

        # How many nodes actually in this row?
        nodes_in_row = min(nodes_per_row, n - row * nodes_per_row)

        # Snake: even rows go left→right, odd rows go right→left.
        # All rows are right-aligned (anchored to column nodes_per_row-1)
        # so the wrap-around connector is always short and local.
        right_anchor = nodes_per_row - 1
        if row % 2 == 0:
            # left→right, but shifted so the row ends at right_anchor
            col = right_anchor - (nodes_in_row - 1 - col)
        else:
            # right→left: col 0 maps to right_anchor, col 1 to right_anchor-1 …
            col = right_anchor - col

        x = col * (box_w + gap_x)
        y = row * (box_h + gap_row)

        positioned.append({**layer, "x": x, "y": y, "row": row,
                            "col": col, "nodes_in_row": nodes_in_row})

    return positioned, num_rows

def auto_edges(network):
    """Simple chain: each node connects to the next."""
    ids = [n["id"] for n in network]
    return [(ids[i], ids[i+1]) for i in range(len(ids)-1)]

# ─────────────────────────────────────────────────────────────────────────────
# SVG renderer
# ─────────────────────────────────────────────────────────────────────────────

def node_center(node, box_w, box_h):
    return node["x"] + box_w / 2, node["y"] + box_h / 2

def render_svg(network, edges, nodes_per_row, box_w, box_h, gap_x, gap_row):
    positioned, num_rows = layout(network, nodes_per_row, box_w, box_h, gap_x, gap_row)
    node_by_id = {n["id"]: n for n in positioned}

    # Canvas size
    max_x = max(n["x"] for n in positioned) + box_w
    max_y = max(n["y"] for n in positioned) + box_h

    legend_h = len(LEGEND_ITEMS) * 22 + 40
    canvas_w = max(max_x + 40, 500)
    canvas_h = max_y + gap_row + legend_h + 20

    lines = []

    # ── Header ──────────────────────────────────────────────────────────────
    lines.append(f'<svg width="100%" viewBox="0 0 {canvas_w} {canvas_h}" '
                 f'xmlns="http://www.w3.org/2000/svg" role="img">')
    lines.append('  <title>Neural network architecture diagram</title>')
    lines.append('  <desc>Layer-by-layer diagram with snake layout</desc>')

    # ── Dark mode style ──────────────────────────────────────────────────────
    style_rules = []
    for typ, dark in DARK_COLORS.items():
        style_rules.append(
            f'    .node-{typ} rect {{ fill:{dark["fill"]}; stroke:{dark["stroke"]}; }}\n'
            f'    .node-{typ} .ntitle {{ fill:{dark["title"]}; }}\n'
            f'    .node-{typ} .nsub   {{ fill:{dark["sub"]}; }}\n'
            f'    .leg-{typ}          {{ fill:{dark["fill"]}; stroke:{dark["stroke"]}; }}\n'
            f'    .leg-lbl-{typ}      {{ fill:{dark["title"]}; }}'
        )
    style_rules.append('    .arr { stroke: #888780; }')
    style_rules.append('    .arr-label { fill: #888780; }')
    lines.append('  <style>')
    lines.append('    @media (prefers-color-scheme: dark) {')
    lines.append('\n'.join(style_rules))
    lines.append('    }')
    lines.append('  </style>')

    # ── Arrow marker ─────────────────────────────────────────────────────────
    lines.append('  <defs>')
    lines.append('    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" '
                 'markerWidth="6" markerHeight="6" orient="auto-start-reverse">')
    lines.append('      <path d="M2 1L8 5L2 9" fill="none" stroke="#888780" '
                 'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>')
    lines.append('    </marker>')
    lines.append('  </defs>')

    # ── Edges ─────────────────────────────────────────────────────────────────
    for src_id, dst_id in edges:
        src = node_by_id[src_id]
        dst = node_by_id[dst_id]
        sx, sy = node_center(src, box_w, box_h)
        dx, dy = node_center(dst, box_w, box_h)

        same_row = src["row"] == dst["row"]

        if same_row:
            # Horizontal arrow: connect whichever side faces the destination
            going_right = dst["x"] > src["x"]
            if going_right:
                x1 = src["x"] + box_w
                x2 = dst["x"]
            else:
                x1 = src["x"]
                x2 = dst["x"] + box_w
            y1 = src["y"] + box_h / 2
            y2 = dst["y"] + box_h / 2
            lines.append(f'  <line x1="{x1:.1f}" y1="{y1:.1f}" '
                         f'x2="{x2:.1f}" y2="{y2:.1f}" '
                         f'stroke="#888780" stroke-width="1.2" '
                         f'marker-end="url(#arrow)"/>')
        else:
            # Between-row: L-shaped path down from src bottom, across, then down to dst top
            x1   = src["x"] + box_w / 2
            y1   = src["y"] + box_h
            ymid = src["y"] + box_h + gap_row / 2
            x2   = dst["x"] + box_w / 2
            y2   = dst["y"]
            path = (f"M{x1:.1f},{y1:.1f} "
                    f"L{x1:.1f},{ymid:.1f} "
                    f"L{x2:.1f},{ymid:.1f} "
                    f"L{x2:.1f},{y2:.1f}")
            lines.append(f'  <path d="{path}" fill="none" '
                         f'stroke="#888780" stroke-width="1.2" '
                         f'marker-end="url(#arrow)"/>')

    # ── Nodes ─────────────────────────────────────────────────────────────────
    for node in positioned:
        typ = node["type"]
        c   = COLORS[typ]
        x, y = node["x"], node["y"]
        cx   = x + box_w / 2

        # Three optional text lines: title (bold), sublabel, output shape
        # Vertical layout: evenly space whichever lines are non-empty
        has_sub = bool(node["sublabel"])
        has_out = bool(node.get("output", ""))
        n_lines = 1 + int(has_sub) + int(has_out)

        # Distribute lines evenly across the box height with 14px spacing
        line_gap = 14
        total_span = (n_lines - 1) * line_gap
        first_y = y + box_h / 2 - total_span / 2

        text_lines = [(node["label"], "title")]
        if has_sub:
            text_lines.append((node["sublabel"], "sub"))
        if has_out:
            text_lines.append((node.get("output", ""), "out"))

        lines.append(f'  <g class="node-{typ}">')
        lines.append(f'    <rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" '
                     f'rx="8" fill="{c["fill"]}" stroke="{c["stroke"]}" stroke-width="0.8"/>')
        for li, (txt, role) in enumerate(text_lines):
            ty = first_y + li * line_gap
            if role == "title":
                lines.append(f'    <text x="{cx:.1f}" y="{ty:.1f}" '
                             f'text-anchor="middle" dominant-baseline="central" '
                             f'font-family="system-ui,sans-serif" font-size="13" '
                             f'font-weight="500" fill="{c["title"]}">{txt}</text>')
            elif role == "sub":
                lines.append(f'    <text x="{cx:.1f}" y="{ty:.1f}" '
                             f'text-anchor="middle" dominant-baseline="central" '
                             f'font-family="system-ui,sans-serif" font-size="11" '
                             f'fill="{c["sub"]}">{txt}</text>')
            else:  # output shape — lighter, italic
                lines.append(f'    <text x="{cx:.1f}" y="{ty:.1f}" '
                             f'text-anchor="middle" dominant-baseline="central" '
                             f'font-family="system-ui,sans-serif" font-size="11" '
                             f'font-style="italic" fill="{c["sub"]}" opacity="0.8">{txt}</text>')
        lines.append('  </g>')

    # ── Legend ────────────────────────────────────────────────────────────────
    leg_y = max_y + gap_row
    lines.append(f'  <text x="0" y="{leg_y}" font-family="system-ui,sans-serif" '
                 f'font-size="11" font-weight="500" fill="#888780">Legend</text>')

    for i, (typ, label) in enumerate(LEGEND_ITEMS):
        c  = COLORS[typ]
        iy = leg_y + 16 + i * 22
        lines.append(f'  <rect class="leg-{typ}" x="0" y="{iy}" width="14" height="14" '
                     f'rx="3" fill="{c["fill"]}" stroke="{c["stroke"]}" stroke-width="0.8"/>')
        lines.append(f'  <text class="leg-lbl-{typ}" x="20" y="{iy+7}" '
                     f'dominant-baseline="central" font-family="system-ui,sans-serif" '
                     f'font-size="11" fill="{c["title"]}">{label}</text>')

    lines.append('</svg>')
    return '\n'.join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("diagram.svg")

    edges = EDGES if EDGES is not None else auto_edges(NETWORK)
    svg   = render_svg(NETWORK, edges, NODES_PER_ROW, BOX_W, BOX_H, GAP_X, GAP_ROW)

    out_path.write_text(svg, encoding="utf-8")
    print(f"SVG written to {out_path}  ({len(NETWORK)} nodes, {len(edges)} edges)")
