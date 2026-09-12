"""
viz_dota.py
===========

Visual inspection of DOTA-format ground truth on high-resolution boards.

Two views, because one is not enough at this resolution:

* :func:`show_board` -- the whole image, downscaled to fit a figure. Good for
  "where are the defects and how many", useless for "is the box tight".
* :func:`show_object_crops` -- a zoomed patch around every annotated object.
  This is the one that actually tells you whether the conversion put boxes on
  the right pixels.

Coordinate convention: DOTA files hold absolute-pixel 4-point polygons,
origin top-left, y down -- the same space OpenCV draws in, so no flipping.

Usage::

    from going_modular.viz_dota import show_split
    show_split(layout, cfg, "test")
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from yolo_obb_to_dota import (  # adjust to `from .yolo_obb_to_dota import ...` if you package it
    ConvertConfig,
    DatasetLayout,
    find_image_for_stem,
    read_dota_file,
)

# Distinct hues; extended automatically if a dataset has more classes.
_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60),
)


def class_color_map(class_names: Sequence[str]) -> Dict[str, Tuple[int, int, int]]:
    """Stable name -> RGB mapping so colours don't shuffle between images."""
    return {name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(sorted(class_names))}


def load_rgb(path: Path) -> np.ndarray:
    """Read an image as RGB (cv2 gives BGR)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw_annotations(
    image: np.ndarray,
    objects: Sequence[Tuple[Sequence[Tuple[float, float]], str, int]],
    colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    thickness: int = 3,
    label: bool = True,
    font_scale: float = 1.2,
) -> np.ndarray:
    """Draw oriented boxes onto a copy of ``image``.

    ``objects`` is what :func:`read_dota_file` returns: ``(points, class_name,
    difficult)`` triples in absolute pixels.
    """
    canvas = image.copy()
    colors = colors or {}
    for points, name, difficult in objects:
        color = colors.get(name, (255, 255, 0))
        poly = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [poly], isClosed=True, color=color,
                      thickness=thickness, lineType=cv2.LINE_AA)
        if label:
            x, y = int(min(p[0] for p in points)), int(min(p[1] for p in points))
            text = f"{name}{'*' if difficult else ''}"
            cv2.putText(canvas, text, (x, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, color, 2, cv2.LINE_AA)
    return canvas


def resolve_paths(
    layout: DatasetLayout, cfg: ConvertConfig, split: str, stem: str
) -> Tuple[Optional[Path], Path]:
    """Locate the image and its converted annotation file for one stem."""
    images_dir = layout.images_dir(split)
    ann_dir = (
        layout.split_dir(split) / cfg.ann_dirname
        if cfg.output_root is None
        else Path(cfg.output_root) / split / cfg.ann_dirname
    )
    image_path = find_image_for_stem(images_dir, stem, cfg.image_extensions)
    return image_path, ann_dir / f"{stem}.txt"


def list_stems(layout: DatasetLayout, cfg: ConvertConfig, split: str) -> List[str]:
    """Every annotated stem in a split, sorted."""
    ann_dir = (
        layout.split_dir(split) / cfg.ann_dirname
        if cfg.output_root is None
        else Path(cfg.output_root) / split / cfg.ann_dirname
    )
    return sorted(p.stem for p in ann_dir.glob("*.txt"))


def show_board(
    layout: DatasetLayout,
    cfg: ConvertConfig,
    split: str,
    stem: str,
    colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    max_side: int = 1400,
    figsize: Tuple[float, float] = (13, 13),
) -> List[Tuple[Sequence[Tuple[float, float]], str, int]]:
    """Show a whole board with its boxes drawn, downscaled to ``max_side``.

    Returns the object list so callers can chain into :func:`show_object_crops`
    without re-reading the file.
    """
    image_path, ann_path = resolve_paths(layout, cfg, split, stem)
    if image_path is None:
        raise FileNotFoundError(f"no image found for stem {stem!r} in split {split!r}")

    image = load_rgb(image_path)
    objects = read_dota_file(ann_path) if ann_path.exists() else []

    # Draw at full resolution, then downscale, so thin lines survive.
    scale = min(1.0, max_side / max(image.shape[:2]))
    thickness = max(2, int(round(4 / scale)))
    font_scale = max(1.0, 2.5 / scale)
    canvas = draw_annotations(image, objects, colors, thickness, True, font_scale)
    if scale < 1.0:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    plt.figure(figsize=figsize)
    plt.imshow(canvas)
    plt.title(f"{split} / {stem}  |  {image.shape[1]}x{image.shape[0]}  |  {len(objects)} objects")
    plt.axis("off")
    plt.show()
    return objects


def show_object_crops(
    layout: DatasetLayout,
    cfg: ConvertConfig,
    split: str,
    stem: str,
    objects: Optional[Sequence] = None,
    context_px: int = 120,
    colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    ncols: int = 5,
    crop_size: float = 2.6,
) -> None:
    """Zoomed patch around each annotated object -- the real check.

    ``context_px`` is the margin added around each box's axis-aligned bounds.
    """
    image_path, ann_path = resolve_paths(layout, cfg, split, stem)
    if image_path is None:
        raise FileNotFoundError(f"no image found for stem {stem!r} in split {split!r}")

    image = load_rgb(image_path)
    if objects is None:
        objects = read_dota_file(ann_path) if ann_path.exists() else []
    if not objects:
        print(f"{stem}: no objects")
        return

    height, width = image.shape[:2]
    nrows = int(np.ceil(len(objects) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(crop_size * ncols, crop_size * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (points, name, difficult) in zip(axes, objects):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x0 = int(max(min(xs) - context_px, 0))
        y0 = int(max(min(ys) - context_px, 0))
        x1 = int(min(max(xs) + context_px, width))
        y1 = int(min(max(ys) + context_px, height))

        patch = image[y0:y1, x0:x1]
        shifted = [((px - x0, py - y0) for px, py in points)]
        shifted_obj = [([(px - x0, py - y0) for px, py in points], name, difficult)]
        patch = draw_annotations(patch, shifted_obj, colors, thickness=2,
                                 label=False, font_scale=0.6)

        ax.imshow(patch)
        ax.set_title(f"{name}\n{int(max(xs) - min(xs))}x{int(max(ys) - min(ys))} px", fontsize=9)
        ax.axis("off")

    for ax in axes[len(objects):]:
        ax.axis("off")

    fig.suptitle(f"{split} / {stem} -- {len(objects)} objects", fontsize=11)
    fig.tight_layout()
    plt.show()


def show_split(
    layout: DatasetLayout,
    cfg: ConvertConfig,
    split: str,
    class_names: Optional[Sequence[str]] = None,
    stems: Optional[Sequence[str]] = None,
    boards: bool = True,
    crops: bool = True,
    context_px: int = 120,
    max_side: int = 1400,
) -> None:
    """Walk a whole split, showing board view and/or per-object crops."""
    stems = list(stems) if stems is not None else list_stems(layout, cfg, split)
    colors = class_color_map(class_names) if class_names else None

    print(f"{split}: {len(stems)} annotated images")
    for stem in stems:
        objects = None
        if boards:
            objects = show_board(layout, cfg, split, stem, colors, max_side)
        if crops:
            show_object_crops(layout, cfg, split, stem, objects, context_px, colors)


def class_size_report(
    layout: DatasetLayout, cfg: ConvertConfig, split: str
) -> "object":
    """Per-object width/height in pixels -- use this to pick tiling overlap.

    Returns a pandas DataFrame if pandas is available, else a list of dicts.
    """
    rows = []
    for stem in list_stems(layout, cfg, split):
        _, ann_path = resolve_paths(layout, cfg, split, stem)
        for points, name, _difficult in read_dota_file(ann_path):
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            rows.append({
                "stem": stem,
                "class": name,
                "width_px": max(xs) - min(xs),
                "height_px": max(ys) - min(ys),
                "max_side_px": max(max(xs) - min(xs), max(ys) - min(ys)),
            })
    try:
        import pandas as pd
        return pd.DataFrame(rows)
    except ImportError:
        return rows
