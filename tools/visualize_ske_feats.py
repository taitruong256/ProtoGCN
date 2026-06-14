"""Render the CASIA-B feature figures as SVG figures.

The feature definitions follow the formulas in
``protogcn/datasets/pipelines/pose_related.py``:

* ``joint``: raw coordinates
* ``bone``: ``joint[i] - joint[parent[i]]``
* ``key-bone``: ``joint[i] - joint[key[i]]``
* ``joint_motion``: ``joint[t + 1] - joint[t]``
* ``bone_motion``: ``bone[t + 1] - bone[t]``
* ``key-bone_motion``: ``key-bone[t + 1] - key-bone[t]``
* ``angle``: joint angle descriptor from ``JointToAngle``
* ``relative``: relative joint offset from ``JointToRelative``

The figures are saved under ``data/figures/features/<sequence>/`` by default.
This script is intentionally dependency-light and writes self-contained SVGs.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from html import escape
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

COCO_KEYPOINTS: Tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# The exact COCO mappings used by GenSkeFeat in pose_related.py.
BONE_PARENT_MAP = {
    0: 0,
    1: 0,
    2: 0,
    3: 1,
    4: 2,
    5: 0,
    6: 0,
    7: 5,
    8: 6,
    9: 7,
    10: 8,
    11: 0,
    12: 0,
    13: 11,
    14: 12,
    15: 13,
    16: 14,
}

KEY_BONE_PARENT_MAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 0,
    4: 0,
    5: 5,
    6: 6,
    7: 0,
    8: 0,
    9: 5,
    10: 6,
    11: 11,
    12: 12,
    13: 0,
    14: 0,
    15: 11,
    16: 12,
}

# The bone tree used for visualization follows graph.py layout="coco".
BONE_TREE_EDGES: Tuple[Tuple[int, int], ...] = (
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 5),
    (12, 6),
    (9, 7),
    (7, 5),
    (10, 8),
    (8, 6),
    (5, 0),
    (6, 0),
    (1, 0),
    (3, 1),
    (2, 0),
    (4, 2),
)

KEY_BONE_TREE_EDGES: Tuple[Tuple[int, int], ...] = tuple(
    (child, parent) for child, parent in KEY_BONE_PARENT_MAP.items() if child != parent
)

COCO_ANGLE_LIST: Tuple[Tuple[int, ...], ...] = (
    (0, 1, 2),
    (1, 0, 3),
    (2, 4, 0),
    (3, 1),
    (4, 2),
    (5, 7, 11),
    (6, 8, 12),
    (7, 5, 9),
    (8, 10, 6),
    (9, 7),
    (10, 8),
    (11, 5, 13),
    (12, 6, 14),
    (13, 11, 15),
    (14, 12, 16),
    (15, 13),
    (16, 14),
)


def _seq_name_from_image_name(image_name: str) -> str:
    seq_name = os.path.normpath(image_name).split(os.sep)[0]
    if seq_name == ".":
        seq_name = os.path.normpath(image_name).split(os.sep)[1]
    return seq_name


def _load_casia_b_sequences(ann_file: str) -> List[Tuple[str, Dict[str, List]]]:
    grouped = defaultdict(list)
    with open(ann_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq_name = _seq_name_from_image_name(row["image_name"])
            frame_id = int(Path(row["image_name"]).stem)

            keypoint = []
            keypoint_score = []
            for name in COCO_KEYPOINTS:
                keypoint.append([float(row[f"{name}_x"]), float(row[f"{name}_y"])])
                keypoint_score.append(float(row[f"{name}_conf"]))

            grouped[seq_name].append((frame_id, keypoint, keypoint_score))

    sequences = []
    for seq_name, frames in sorted(grouped.items()):
        frames = sorted(frames, key=lambda x: x[0])
        _, keypoints, scores = zip(*frames)
        sequences.append(
            (
                seq_name,
                {
                    "keypoint": [list(frame) for frame in keypoints],
                    "keypoint_score": [list(frame) for frame in scores],
                },
            )
        )
    return sequences


def _select_sequence(
    sequences: List[Tuple[str, Dict[str, List]]],
    seq_name: Optional[str],
    sample_index: int,
) -> Tuple[str, Dict[str, List]]:
    if not sequences:
        raise ValueError("No sequences found in the annotation file.")

    if seq_name is not None:
        for name, payload in sequences:
            if name == seq_name:
                return name, payload
        available = ", ".join(name for name, _ in sequences[:10])
        raise KeyError(f"Sequence '{seq_name}' was not found. Available examples: {available}")

    if sample_index < 0 or sample_index >= len(sequences):
        raise IndexError(f"sample-index {sample_index} is out of range for {len(sequences)} sequences")

    return sequences[sample_index]


def _vector_sub(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [float(a[0]) - float(b[0]), float(a[1]) - float(b[1])]


def _compute_relative_frames(frames: List[List[List[float]]]) -> np.ndarray:
    coords = np.asarray(frames, dtype=np.float32)
    root = coords[:, :1, :]
    return coords - root


def _cos_law(center: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    side1 = np.sqrt((center[:, 0] - left[:, 0]) ** 2 + (center[:, 1] - left[:, 1]) ** 2)
    side2 = np.sqrt((center[:, 0] - right[:, 0]) ** 2 + (center[:, 1] - right[:, 1]) ** 2)
    side3 = np.sqrt((left[:, 0] - right[:, 0]) ** 2 + (left[:, 1] - right[:, 1]) ** 2)
    deno = side1 * side2
    where_zero = deno == 0
    deno = deno.copy()
    deno[where_zero] = 1.0
    cos = (side1 * side1 + side2 * side2 - side3 * side3) / (2 * deno)
    cos = np.clip(cos, -1.0, 1.0)
    value = np.pi - np.arccos(cos)
    value[where_zero] = np.pi
    return np.nan_to_num(value).astype(np.float32)


def _compute_angle_frames(frames: List[List[List[float]]]) -> np.ndarray:
    coords = np.asarray(frames, dtype=np.float32)
    if coords.shape[1] != len(COCO_ANGLE_LIST):
        raise ValueError(
            f"Angle feature expects {len(COCO_ANGLE_LIST)} joints for coco, "
            f"but got {coords.shape[1]}."
        )

    angle = np.zeros((coords.shape[0], coords.shape[1]), dtype=np.float32)
    for i, angle_def in enumerate(COCO_ANGLE_LIST):
        if len(angle_def) == 3:
            center = coords[:, angle_def[0], :]
            left = coords[:, angle_def[1], :]
            right = coords[:, angle_def[2], :]
            angle[:, i] = _cos_law(center, left, right)
        else:
            center = coords[:, angle_def[0], :]
            left = coords[:, angle_def[1], :]
            right = np.zeros_like(center)
            right[:, 0] = center[:, 0]
            right[:, 1] = left[:, 1]
            angle[:, i] = _cos_law(center, left, right)
    return angle


def _compute_feature_frames(frames: List[List[List[float]]], parent_map: Dict[int, int]) -> List[List[List[float]]]:
    out = []
    for frame in frames:
        feat_frame = []
        for idx, point in enumerate(frame):
            parent = parent_map[idx]
            feat_frame.append(_vector_sub(point, frame[parent]))
        out.append(feat_frame)
    return out


def _frame(frames: List[List[List[float]]], index: int) -> List[List[float]]:
    return frames[index]


def _scores_frame(scores: List[List[float]], index: int) -> List[float]:
    return scores[index]


def _edge_midpoints(points: List[List[float]], edges: Iterable[Tuple[int, int]]) -> List[List[float]]:
    mids = []
    for child, parent in edges:
        mids.append([
            (float(points[child][0]) + float(points[parent][0])) / 2.0,
            (float(points[child][1]) + float(points[parent][1])) / 2.0,
        ])
    return mids


def _append_xy(obj, out: List[Tuple[float, float]]):
    if isinstance(obj, (list, tuple)):
        if not obj:
            return
        if isinstance(obj[0], (int, float)):
            if len(obj) >= 2:
                out.append((float(obj[0]), float(obj[1])))
            return
        for item in obj:
            _append_xy(item, out)


def _plot_limits(*arrays, pad: float = 0.15) -> Tuple[float, float, float, float]:
    coords: List[Tuple[float, float]] = []
    for arr in arrays:
        _append_xy(arr, coords)

    if not coords:
        return -1.0, 1.0, -1.0, 1.0

    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    width = max(xmax - xmin, 1e-6)
    height = max(ymax - ymin, 1e-6)
    margin = max(width, height) * pad
    return xmin - margin, xmax + margin, ymin - margin, ymax + margin


def _joint_color(idx: int) -> str:
    palette = [
        "#2E86AB",
        "#F6AE2D",
        "#F26419",
        "#33658A",
        "#55DDE0",
        "#7E3F8F",
    ]
    return palette[idx % len(palette)]


def _svg_escape(text: str) -> str:
    return escape(str(text), quote=True)


def _fmt(value: float) -> str:
    return f"{float(value):.2f}"


def _make_mapper(
    bounds: Tuple[float, float, float, float],
    box: Tuple[float, float, float, float],
    pad: float = 34.0,
):
    xmin, xmax, ymin, ymax = bounds
    x0, y0, w, h = box
    left = x0 + pad
    right = x0 + w - pad
    top = y0 + pad
    bottom = y0 + h - pad
    sx = max(xmax - xmin, 1e-6)
    sy = max(ymax - ymin, 1e-6)

    def map_point(point: Sequence[float]) -> Tuple[float, float]:
        x, y = float(point[0]), float(point[1])
        px = left + ((x - xmin) / sx) * (right - left)
        py = top + ((y - ymin) / sy) * (bottom - top)
        return px, py

    return map_point


def _svg_header(width: int, height: int, title: str, transparent_bg: bool = False) -> List[str]:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<marker id="arrow-red" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto" markerUnits="strokeWidth">',
        '<path d="M0,0 L8,4 L0,8 z" fill="#D1495B" />',
        "</marker>",
        "</defs>",
    ]
    if not transparent_bg:
        lines.append(
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="#FBFCFE" />'
        )
        lines.append(
            f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" font-size="22" font-family="DejaVu Sans, Arial, sans-serif" font-weight="700" fill="#152238">{_svg_escape(title)}</text>'
        )
    else:
        lines.append('<rect x="0" y="0" width="100%" height="100%" fill="none" />')
    return lines


def _svg_footer() -> List[str]:
    return ["</svg>"]


def _panel_box(index: int, total: int, canvas_w: int, canvas_h: int, top: int = 62, bottom_pad: int = 24, gap: int = 22):
    outer_w = canvas_w - 2 * 22 - gap * (total - 1)
    panel_w = outer_w / total
    panel_h = canvas_h - top - bottom_pad
    x = 22 + index * (panel_w + gap)
    y = top
    return x, y, panel_w, panel_h


def _rect(x: float, y: float, w: float, h: float, fill: str = "none", stroke: str = "#D8DEE9", stroke_width: float = 1.5, rx: float = 14.0) -> str:
    return (
        f'<rect x="{_fmt(x)}" y="{_fmt(y)}" width="{_fmt(w)}" height="{_fmt(h)}" '
        f'rx="{_fmt(rx)}" ry="{_fmt(rx)}" fill="{fill}" stroke="{stroke}" stroke-width="{_fmt(stroke_width)}" />'
    )


def _text(x: float, y: float, value: str, size: int = 15, weight: str = "600", fill: str = "#152238", anchor: str = "middle") -> str:
    return (
        f'<text x="{_fmt(x)}" y="{_fmt(y)}" text-anchor="{anchor}" '
        f'font-size="{size}" font-family="DejaVu Sans, Arial, sans-serif" font-weight="{weight}" fill="{fill}">{_svg_escape(value)}</text>'
    )


def _circle(x: float, y: float, r: float = 6.5, fill: str = "#2E86AB", opacity: float = 1.0, stroke: str = "#FFFFFF") -> str:
    return (
        f'<circle cx="{_fmt(x)}" cy="{_fmt(y)}" r="{_fmt(r)}" fill="{fill}" '
        f'fill-opacity="{_fmt(opacity)}" stroke="{stroke}" stroke-width="1.2" />'
    )


def _line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str = "#222222",
    width: float = 3.5,
    opacity: float = 0.9,
    dash: Optional[str] = None,
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{_fmt(x1)}" y1="{_fmt(y1)}" x2="{_fmt(x2)}" y2="{_fmt(y2)}" '
        f'stroke="{color}" stroke-width="{_fmt(width)}" stroke-linecap="round" stroke-opacity="{_fmt(opacity)}"{dash_attr} />'
    )


def _path(d: str, fill: str = "none", stroke: str = "#222222", width: float = 2.5, opacity: float = 0.9, dash: Optional[str] = None) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{_fmt(width)}" '
        f'stroke-linecap="round" stroke-linejoin="round" stroke-opacity="{_fmt(opacity)}"{dash_attr} />'
    )


def _arrow(x1: float, y1: float, x2: float, y2: float, color: str = "#D1495B", width: float = 2.0, opacity: float = 0.95) -> List[str]:
    return [
        f'<line x1="{_fmt(x1)}" y1="{_fmt(y1)}" x2="{_fmt(x2)}" y2="{_fmt(y2)}" '
        f'stroke="{color}" stroke-width="{_fmt(width)}" stroke-linecap="round" stroke-opacity="{_fmt(opacity)}" marker-end="url(#arrow-red)" />',
    ]


def _arc_path(center: Tuple[float, float], start: Tuple[float, float], end: Tuple[float, float], radius_scale: float = 0.34) -> str:
    cx, cy = center
    sx, sy = start
    ex, ey = end
    theta1 = math.atan2(sy - cy, sx - cx)
    theta2 = math.atan2(ey - cy, ex - cx)
    r1 = math.hypot(sx - cx, sy - cy)
    r2 = math.hypot(ex - cx, ey - cy)
    radius = max(8.0, min(r1, r2) * radius_scale)

    x1 = cx + radius * math.cos(theta1)
    y1 = cy + radius * math.sin(theta1)
    x2 = cx + radius * math.cos(theta2)
    y2 = cy + radius * math.sin(theta2)

    delta = theta2 - theta1
    while delta <= -math.pi:
        delta += 2 * math.pi
    while delta > math.pi:
        delta -= 2 * math.pi

    large_arc = 1 if abs(delta) > math.pi else 0
    sweep = 1 if delta > 0 else 0
    return f"M {_fmt(x1)} {_fmt(y1)} A {_fmt(radius)} {_fmt(radius)} 0 {large_arc} {sweep} {_fmt(x2)} {_fmt(y2)}"


def _draw_joints(points: List[List[float]], scores: Optional[List[float]], map_point, color: str, label: bool = True) -> List[str]:
    svg = []
    if scores is None:
        alphas = [1.0 for _ in points]
    else:
        alphas = [max(0.15, min(1.0, float(score))) for score in scores]

    for idx, point in enumerate(points):
        px, py = map_point(point)
        svg.append(_circle(px, py, r=6.5, fill=color, opacity=float(alphas[idx])))
        if label:
            svg.append(
                f'<text x="{_fmt(px)}" y="{_fmt(py - 8)}" text-anchor="middle" font-size="10" '
                f'font-family="DejaVu Sans, Arial, sans-serif" fill="#152238">{idx}</text>'
            )
    return svg


def _draw_edges(
    points: List[List[float]],
    edges: Iterable[Tuple[int, int]],
    scores: Optional[List[float]],
    map_point,
    color: str = "#202020",
) -> List[str]:
    svg = []
    for child, parent in edges:
        if child == parent:
            continue
        x1, y1 = points[child]
        x2, y2 = points[parent]
        if scores is not None:
            opacity = max(0.12, min(1.0, (float(scores[child]) + float(scores[parent])) / 2.0))
        else:
            opacity = 0.85
        px1, py1 = map_point((x1, y1))
        px2, py2 = map_point((x2, y2))
        svg.append(_line(px1, py1, px2, py2, color=color, opacity=opacity))
    return svg


def _edge_midpoints(points: List[List[float]], edges: Iterable[Tuple[int, int]]) -> List[List[float]]:
    mids = []
    for child, parent in edges:
        mids.append([
            (float(points[child][0]) + float(points[parent][0])) / 2.0,
            (float(points[child][1]) + float(points[parent][1])) / 2.0,
        ])
    return mids


def _write_svg(path: Path, width: int, height: int, title: str, body: List[str], transparent_bg: bool = False):
    lines = _svg_header(width, height, title, transparent_bg=transparent_bg)
    lines.extend(body)
    lines.extend(_svg_footer())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_relative_figure(
    frame: List[List[float]],
    output_file: Path,
    scores: Optional[List[float]] = None,
    title: str = "Relative feature: J_i to J_0",
):
    """Draw solid connections from root joint J_0 to every other joint."""
    canvas_w, canvas_h = 760, 760
    box = _panel_box(0, 1, canvas_w, canvas_h)
    bounds = _plot_limits(frame)
    map_point = _make_mapper(bounds, box)

    x, y, w, h = box
    body = [
        _rect(x, y, w, h),
        _text(x + w / 2, y + 28, title, size=17),
    ]
    root = frame[0]
    rx, ry = map_point(root)
    body.append(_circle(rx, ry, r=8.5, fill="#D1495B", opacity=1.0))
    body.append(_text(rx + 16, ry - 10, "J0", size=11, weight="700", fill="#D1495B", anchor="start"))

    for idx, joint in enumerate(frame):
        if idx == 0:
            continue
        jx, jy = map_point(joint)
        opacity = 0.95
        if scores is not None:
            opacity = max(0.2, min(1.0, float(scores[idx])))
        body.append(_line(rx, ry, jx, jy, color="#2E86AB", width=2.8, opacity=opacity))
        body.append(_circle(jx, jy, r=5.5, fill="#2E86AB", opacity=0.95))
        body.append(_text(jx + 10, jy - 8, str(idx), size=9, weight="600", fill="#152238", anchor="start"))

    _write_svg(output_file, canvas_w, canvas_h, f"{title}", body)


def _save_angle_figure(
    frame: List[List[float]],
    output_file: Path,
    scores: Optional[List[float]] = None,
    title: str = "Angle feature: arc annotations",
):
    """Draw dashed angle arcs for the coco angle list from pose_related.py."""
    canvas_w, canvas_h = 760, 760
    box = _panel_box(0, 1, canvas_w, canvas_h)
    bounds = _plot_limits(frame)
    map_point = _make_mapper(bounds, box)

    x, y, w, h = box
    body = [
        _rect(x, y, w, h),
        _text(x + w / 2, y + 28, title, size=17),
    ]
    body.extend(_draw_edges(frame, BONE_TREE_EDGES, scores, map_point, color="#8A8A8A"))
    body.extend(_draw_joints(frame, scores, map_point, color="#6AAE75", label=False))

    for angle_idx, angle_def in enumerate(COCO_ANGLE_LIST):
        color = "#F26419"
        if len(angle_def) == 3:
            center_idx, left_idx, right_idx = angle_def
            center = map_point(frame[center_idx])
            left = map_point(frame[left_idx])
            right = map_point(frame[right_idx])
            body.append(_line(center[0], center[1], left[0], left[1], color=color, width=1.7, opacity=0.7, dash="4,4"))
            body.append(_line(center[0], center[1], right[0], right[1], color=color, width=1.7, opacity=0.7, dash="4,4"))
            body.append(_path(_arc_path(center, left, right), stroke=color, width=2.3, opacity=0.95, dash="4,4"))
            body.append(_text(center[0] + 10, center[1] - 10, str(angle_idx), size=9, weight="700", fill=color, anchor="start"))
        else:
            center_idx, left_idx = angle_def
            center = map_point(frame[center_idx])
            left = map_point(frame[left_idx])
            vertical = (center[0], left[1])
            body.append(_line(center[0], center[1], left[0], left[1], color=color, width=1.7, opacity=0.7))
            body.append(_line(center[0], center[1], vertical[0], vertical[1], color=color, width=1.7, opacity=0.7, dash="4,4"))
            body.append(_path(_arc_path(center, left, vertical), stroke=color, width=2.3, opacity=0.95, dash="4,4"))
            body.append(_text(center[0] + 10, center[1] - 10, str(angle_idx), size=9, weight="700", fill=color, anchor="start"))

    _write_svg(output_file, canvas_w, canvas_h, f"{title}", body)


def _render_joint_figure(seq_name: str, raw_frames: List[List[List[float]]], raw_scores: List[List[float]], output_file: Path):
    canvas_w, canvas_h = 760, 760
    box = _panel_box(0, 1, canvas_w, canvas_h)
    frame0 = _frame(raw_frames, 0)
    scores0 = _scores_frame(raw_scores, 0)
    bounds = _plot_limits(frame0)
    map_point = _make_mapper(bounds, box)

    x, y, w, h = box
    body = [
        _rect(x, y, w, h),
        _text(x + w / 2, y + 28, "Joint: first frame raw coordinates", size=17),
    ]
    body.extend(_draw_joints(frame0, scores0, map_point, color="#2E86AB"))
    _write_svg(output_file, canvas_w, canvas_h, f"{seq_name} | joint", body)


def _render_tree_figure(
    seq_name: str,
    raw_frames: List[List[List[float]]],
    raw_scores: List[List[float]],
    edges: Iterable[Tuple[int, int]],
    label: str,
    output_file: Path,
):
    canvas_w, canvas_h = 760, 760
    box = _panel_box(0, 1, canvas_w, canvas_h)
    frame0 = _frame(raw_frames, 0)
    scores0 = _scores_frame(raw_scores, 0)
    bounds = _plot_limits(frame0)
    map_point = _make_mapper(bounds, box)

    x, y, w, h = box
    body = [
        _rect(x, y, w, h),
        _text(x + w / 2, y + 28, label, size=17),
    ]
    body.extend(_draw_edges(frame0, edges, scores0, map_point, color="#252A34"))
    body.extend(_draw_joints(frame0, scores0, map_point, color="#2E86AB"))
    _write_svg(output_file, canvas_w, canvas_h, f"{seq_name} | {label.split(':', 1)[0].lower()}", body)


def _render_frame_figure(
    seq_name: str,
    raw_frames: List[List[List[float]]],
    raw_scores: List[List[float]],
    frame_idx: int,
    output_file: Path,
):
    canvas_w, canvas_h = 760, 760
    box = _panel_box(0, 1, canvas_w, canvas_h)
    frame = _frame(raw_frames, frame_idx)
    scores = _scores_frame(raw_scores, frame_idx)
    bounds = _plot_limits(frame)
    map_point = _make_mapper(bounds, box)

    body = [
    ]
    body.extend(_draw_edges(frame, BONE_TREE_EDGES, scores, map_point, color="#252A34"))
    body.extend(_draw_joints(frame, scores, map_point, color="#2E86AB", label=False))
    _write_svg(output_file, canvas_w, canvas_h, f"{seq_name} | frame {frame_idx}", body, transparent_bg=True)


def _render_motion_figure(
    seq_name: str,
    frame_a: List[List[float]],
    frame_b: List[List[float]],
    scores_a: Optional[List[float]],
    scores_b: Optional[List[float]],
    motion_anchors: List[List[float]],
    motion_deltas: List[List[float]],
    title: str,
    output_file: Path,
    tree_edges: Iterable[Tuple[int, int]],
    draw_tree_on_frames: bool = True,
):
    canvas_w, canvas_h = 1860, 760
    top = 62
    panel_boxes = [_panel_box(i, 3, canvas_w, canvas_h, top=top) for i in range(3)]

    limits = _plot_limits(frame_a, frame_b, motion_anchors, [[a[0] + d[0], a[1] + d[1]] for a, d in zip(motion_anchors, motion_deltas)])
    body: List[str] = []

    for x, y, w, h in panel_boxes:
        body.append(_rect(x, y, w, h))

    # Panel 1
    x, y, w, h = panel_boxes[0]
    map0 = _make_mapper(limits, panel_boxes[0])
    body.append(_text(x + w / 2, y + 28, "Frame 0", size=17))
    body.extend(_draw_joints(frame_a, scores_a, map0, color="#2E86AB"))
    if draw_tree_on_frames:
        body.extend(_draw_edges(frame_a, tree_edges, scores_a, map0, color="#3A3A3A"))

    # Panel 2
    x, y, w, h = panel_boxes[1]
    map1 = _make_mapper(limits, panel_boxes[1])
    body.append(_text(x + w / 2, y + 28, "Frame 5", size=17))
    body.extend(_draw_joints(frame_b, scores_b, map1, color="#F26419"))
    if draw_tree_on_frames:
        body.extend(_draw_edges(frame_b, tree_edges, scores_b, map1, color="#3A3A3A"))

    # Panel 3
    x, y, w, h = panel_boxes[2]
    map2 = _make_mapper(limits, panel_boxes[2])
    body.append(_text(x + w / 2, y + 28, "Motion vectors", size=17))
    for anchor, delta in zip(motion_anchors, motion_deltas):
        ax, ay = anchor
        dx, dy = delta
        sx, sy = map2((ax, ay))
        ex, ey = map2((ax + dx, ay + dy))
        body.extend(_arrow(sx, sy, ex, ey))
        body.append(_circle(sx, sy, r=6.0, fill="#D1495B", opacity=0.95))

    _write_svg(output_file, canvas_w, canvas_h, f"{seq_name} | {title}", body)


def parse_args():
    parser = argparse.ArgumentParser(description="Render the six skeleton features used by the ensemble baseline.")
    parser.add_argument(
        "--ann-file",
        default="data/casia-b/casia-b_pose_valid.csv",
        help="CASIA-B pose CSV annotation file.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/figures/features",
        help="Directory where the feature figures will be written.",
    )
    parser.add_argument(
        "--sequence",
        default=None,
        help="Optional sequence name to visualize, e.g. 001-nm-01-000.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Fallback sequence index if --sequence is not provided.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sequences = _load_casia_b_sequences(args.ann_file)
    seq_name, payload = _select_sequence(sequences, args.sequence, args.sample_index)

    raw_frames = payload["keypoint"]
    raw_scores = payload["keypoint_score"]
    if len(raw_frames) < 6:
        raise ValueError("The selected sequence needs at least six frames for the 0-to-5 motion figures.")
    if len(raw_frames) <= 40:
        raise ValueError("The selected sequence needs at least 41 frames for the 0, 10, 20, 30, 40 figures.")

    output_dir = Path(args.output_dir) / seq_name
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = Path("data/casia-b/figures") / seq_name
    frame_dir.mkdir(parents=True, exist_ok=True)

    print(f"Selected sequence: {seq_name}")
    print(f"Frames: {len(raw_frames)}, joints: {len(raw_frames[0])}")
    print(f"Writing figures to: {output_dir}")

    bone_frames = _compute_feature_frames(raw_frames, BONE_PARENT_MAP)
    key_bone_frames = _compute_feature_frames(raw_frames, KEY_BONE_PARENT_MAP)
    frame0 = _frame(raw_frames, 0)
    frame5 = _frame(raw_frames, 5)
    scores0 = _scores_frame(raw_scores, 0)
    scores5 = _scores_frame(raw_scores, 5)

    joint_motion_anchors = frame0
    joint_motion_deltas = [_vector_sub(frame5[i], frame0[i]) for i in range(len(frame0))]
    bone_motion_anchors = _edge_midpoints(frame0, BONE_TREE_EDGES)
    bone_motion_deltas = [
        _vector_sub(bone_frames[5][child], bone_frames[0][child]) for child, _ in BONE_TREE_EDGES
    ]
    key_bone_motion_anchors = _edge_midpoints(frame0, KEY_BONE_TREE_EDGES)
    key_bone_motion_deltas = [
        _vector_sub(key_bone_frames[5][child], key_bone_frames[0][child]) for child, _ in KEY_BONE_TREE_EDGES
    ]

    _render_joint_figure(seq_name, raw_frames, raw_scores, output_dir / "joint.svg")
    _render_tree_figure(seq_name, raw_frames, raw_scores, BONE_TREE_EDGES, "Bone: first frame skeleton", output_dir / "bone.svg")
    _render_tree_figure(
        seq_name,
        raw_frames,
        raw_scores,
        KEY_BONE_TREE_EDGES,
        "Key-bone: first frame skeleton",
        output_dir / "key-bone.svg",
    )
    _render_motion_figure(
        seq_name,
        frame0,
        frame5,
        scores0,
        scores5,
        joint_motion_anchors,
        joint_motion_deltas,
        "Joint motion",
        output_dir / "joint_motion.svg",
        tree_edges=BONE_TREE_EDGES,
        draw_tree_on_frames=False,
    )
    _render_motion_figure(
        seq_name,
        frame0,
        frame5,
        scores0,
        scores5,
        bone_motion_anchors,
        bone_motion_deltas,
        "Bone motion",
        output_dir / "bone_motion.svg",
        tree_edges=BONE_TREE_EDGES,
    )
    _render_motion_figure(
        seq_name,
        frame0,
        frame5,
        scores0,
        scores5,
        key_bone_motion_anchors,
        key_bone_motion_deltas,
        "Key-bone motion",
        output_dir / "key-bone_motion.svg",
        tree_edges=KEY_BONE_TREE_EDGES,
    )

    _save_relative_figure(
        frame0,
        output_dir / "relative.svg",
        scores=scores0,
        title="Relative feature: solid links from J0 to every joint",
    )
    _save_angle_figure(
        frame0,
        output_dir / "angle.svg",
        scores=scores0,
        title="Angle feature",
    )

    for frame_idx in (0, 10, 20, 30, 40):
        _render_frame_figure(
            seq_name,
            raw_frames,
            raw_scores,
            frame_idx,
            frame_dir / f"frame_{frame_idx:02d}.svg",
        )

    print("Done. Feature figures were written.")
    print(f"Extra frame figures were written to: {frame_dir}")


if __name__ == "__main__":
    main()
