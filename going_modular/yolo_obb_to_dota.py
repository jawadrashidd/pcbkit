"""
yolo_obb_to_dota.py
===================

Convert Ultralytics / Roboflow **YOLO-OBB** label files into **DOTA**-style
annotation files, leaving the original ``labels/`` directory untouched.

Coordinate conventions (read this before debugging anything)
------------------------------------------------------------
INPUT  -- YOLO-OBB, one object per line::

    <class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>

    * 9 whitespace-separated tokens
    * class_id is an integer index into ``names`` from data.yaml
    * all 8 coordinates are NORMALISED to [0, 1] w.r.t. image width/height
    * origin is top-left, y increases downwards
    * vertices are in order (clockwise for Roboflow exports, but this
      converter never relies on winding)

OUTPUT -- DOTA v1.0, one object per line::

    <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4> <class_name> <difficult>

    * 10 whitespace-separated tokens
    * coordinates are ABSOLUTE PIXELS in the source image
    * class_name is a string (spaces are replaced with '-', since the
      format is space-delimited)
    * difficult is 0 or 1

Because YOLO-OBB stores normalised coordinates and DOTA stores pixels, the
image dimensions are required. This module reads them from the image header
via PIL (no full decode), so conversion stays fast on 2600x2500 boards.

Layout assumptions
------------------
The default layout is the Roboflow export shape::

    <root>/
        data.yaml
        train/  images/  labels/
        valid/  images/  labels/
        test/   images/  labels/

Every part of that is overridable: the split folder names, the images folder
name, the labels folder name -- globally or per split, by name or by absolute
path. See :class:`DatasetLayout`.

Nothing is ever deleted or modified in the source tree. Converted files are
written to ``<output_root>/<split>/<ann_dirname>/`` (default ``ann``), plus a
JSON manifest describing the conversion.

Typical use
-----------
::

    from yolo_obb_to_dota import DatasetLayout, ConvertConfig, convert_dataset

    layout = DatasetLayout(root="/kaggle/input/.../PCB-Defect.v3-50_trial_1.yolov8-obb")
    cfg = ConvertConfig(output_root="/kaggle/working/pcb-dota")
    manifest = convert_dataset(layout, cfg)

Command line::

    python yolo_obb_to_dota.py --root <dataset_root> --output /kaggle/working/pcb-dota
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__version__ = "0.1.0"

logger = logging.getLogger(__name__)

Point = Tuple[float, float]
Quad = Tuple[Point, Point, Point, Point]

DEFAULT_IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
)


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrientedBox:
    """A 4-point oriented box.

    ``points`` are in whatever space the producer used -- normalised [0, 1]
    when parsed straight out of a YOLO-OBB file, absolute pixels after
    :meth:`to_pixels`. The class does not track which; callers do.
    """

    points: Quad
    class_id: int
    difficult: int = 0

    def to_pixels(self, width: int, height: int) -> "OrientedBox":
        """Scale normalised coordinates to absolute pixels."""
        scaled = tuple((x * width, y * height) for x, y in self.points)
        return OrientedBox(scaled, self.class_id, self.difficult)  # type: ignore[arg-type]

    def clipped(self, width: float, height: float) -> "OrientedBox":
        """Clamp each vertex into the image rectangle.

        This is a per-vertex clamp, not a polygon intersection -- it fixes
        annotations that spill a few pixels past the border without changing
        the shape of anything fully inside. Use a real polygon clip
        (shapely) if you need to cut boxes at tile boundaries.
        """
        clamped = tuple(
            (min(max(x, 0.0), width), min(max(y, 0.0), height))
            for x, y in self.points
        )
        return OrientedBox(clamped, self.class_id, self.difficult)  # type: ignore[arg-type]

    @property
    def area(self) -> float:
        """Shoelace area; always non-negative, so winding doesn't matter."""
        pts = self.points
        total = 0.0
        for i in range(len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % len(pts)]
            total += x1 * y2 - x2 * y1
        return abs(total) / 2.0

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Axis-aligned (xmin, ymin, xmax, ymax)."""
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return min(xs), min(ys), max(xs), max(ys)


def sanitize_class_name(name: str) -> str:
    """DOTA lines are space-delimited, so class names cannot contain spaces."""
    return "-".join(str(name).split())


# ---------------------------------------------------------------------------
# Parsing / formatting
# ---------------------------------------------------------------------------

class LabelParseError(ValueError):
    """Raised for a malformed YOLO-OBB line."""


def parse_yolo_obb_line(line: str) -> OrientedBox:
    """Parse one YOLO-OBB line into an :class:`OrientedBox` (normalised space).

    Raises :class:`LabelParseError` if the line does not have exactly 9
    numeric tokens.
    """
    tokens = line.split()
    if len(tokens) != 9:
        raise LabelParseError(
            f"expected 9 tokens (class + 4 xy pairs), got {len(tokens)}: {line.strip()!r}"
        )
    try:
        class_id = int(float(tokens[0]))
        coords = [float(t) for t in tokens[1:]]
    except ValueError as exc:
        raise LabelParseError(f"non-numeric token in {line.strip()!r}") from exc

    points = tuple((coords[i], coords[i + 1]) for i in range(0, 8, 2))
    return OrientedBox(points, class_id)  # type: ignore[arg-type]


def read_yolo_obb_file(path: Path) -> Tuple[List[OrientedBox], List[str]]:
    """Read a YOLO-OBB ``.txt``.

    Returns ``(boxes, warnings)``. Blank lines and ``#`` comments are skipped
    silently; malformed lines are skipped with a warning string rather than
    aborting the whole dataset conversion.
    """
    boxes: List[OrientedBox] = []
    warnings: List[str] = []

    text = path.read_text(encoding="utf-8", errors="replace")
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            boxes.append(parse_yolo_obb_line(line))
        except LabelParseError as exc:
            warnings.append(f"{path.name}:{lineno}: {exc}")
    return boxes, warnings


def format_dota_line(
    box: OrientedBox,
    class_name: str,
    decimals: int = 1,
) -> str:
    """Format one pixel-space box as a DOTA annotation line."""
    if decimals <= 0:
        coords = " ".join(f"{int(round(v))}" for pt in box.points for v in pt)
    else:
        coords = " ".join(f"{v:.{decimals}f}" for pt in box.points for v in pt)
    return f"{coords} {sanitize_class_name(class_name)} {int(box.difficult)}"


def read_dota_file(path: Path) -> List[Tuple[Quad, str, int]]:
    """Read a DOTA ``.txt`` back in -- used for round-trip verification.

    Returns a list of ``(points, class_name, difficult)``. Header lines
    (``imagesource:``, ``gsd:``) are ignored.
    """
    out: List[Tuple[Quad, str, int]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith(("imagesource", "gsd", "#")):
            continue
        tokens = line.split()
        if len(tokens) < 9:
            continue
        coords = [float(t) for t in tokens[:8]]
        points = tuple((coords[i], coords[i + 1]) for i in range(0, 8, 2))
        class_name = tokens[8]
        difficult = int(tokens[9]) if len(tokens) > 9 else 0
        out.append((points, class_name, difficult))  # type: ignore[arg-type]
    return out


# ---------------------------------------------------------------------------
# Image + class-name helpers
# ---------------------------------------------------------------------------

def read_image_size(path: Path) -> Tuple[int, int]:
    """Return ``(width, height)`` by reading the image header only.

    PIL is lazy: ``.size`` is available without decoding pixel data, which
    matters when the boards are ~2600x2500.
    """
    try:
        from PIL import Image  # noqa: WPS433 (import kept local so PIL stays optional)
    except ImportError:  # pragma: no cover - fallback path
        import cv2  # type: ignore

        img = cv2.imread(str(path))
        if img is None:
            raise OSError(f"could not read image: {path}")
        return img.shape[1], img.shape[0]

    with Image.open(path) as img:
        return img.size  # (width, height)


def find_image_for_stem(
    images_dir: Path,
    stem: str,
    extensions: Sequence[str] = DEFAULT_IMAGE_EXTENSIONS,
) -> Optional[Path]:
    """Locate the image whose filename stem matches a label file's stem."""
    for ext in extensions:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    # Case-insensitive fallback (some exports mix .JPG / .jpg).
    lowered = {ext.lower() for ext in extensions}
    for candidate in images_dir.glob(f"{stem}.*"):
        if candidate.suffix.lower() in lowered:
            return candidate
    return None


def load_class_names(data_yaml: Path) -> List[str]:
    """Read the ``names`` field out of a Roboflow/Ultralytics ``data.yaml``.

    Handles both the list form (``names: ['a', 'b']``) and the dict form
    (``names: {0: a, 1: b}``), returning a list indexed by class id.
    """
    import yaml  # noqa: WPS433

    with data_yaml.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    names = data.get("names")
    if names is None:
        raise KeyError(f"no 'names' key in {data_yaml}")
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda k: int(k))]
    if isinstance(names, (list, tuple)):
        return [str(n) for n in names]
    raise TypeError(f"unsupported 'names' type in {data_yaml}: {type(names)!r}")


def resolve_class_name(class_names: Sequence[str], class_id: int) -> str:
    """Map an id to a name, falling back to ``class_<id>`` when out of range."""
    if 0 <= class_id < len(class_names):
        return class_names[class_id]
    return f"class_{class_id}"


# ---------------------------------------------------------------------------
# Layout + configuration
# ---------------------------------------------------------------------------

@dataclass
class DatasetLayout:
    """Where the source dataset lives and what its folders are called.

    Every name can be overridden globally or per split. A per-split override
    may be a plain folder name (resolved relative to that split's directory)
    or an absolute path (used as-is), which covers datasets whose splits are
    scattered across different roots.

    Examples
    --------
    Default Roboflow shape::

        DatasetLayout(root="/kaggle/input/.../PCB-Defect.v3-50_trial_1.yolov8-obb")

    Split folders named differently::

        DatasetLayout(root=..., split_dirs={"valid": "validation", "test": "testing"})

    One split's labels living under a different folder name::

        DatasetLayout(root=..., labels_dirs={"train": "annotations"})

    Labels stored somewhere else entirely::

        DatasetLayout(root=..., labels_dirs={"train": "/data/exported/train_labels"})
    """

    root: Path
    splits: Sequence[str] = ("train", "valid", "test")
    images_dirname: str = "images"
    labels_dirname: str = "labels"
    split_dirs: Mapping[str, str] = field(default_factory=dict)
    images_dirs: Mapping[str, str] = field(default_factory=dict)
    labels_dirs: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.splits = tuple(self.splits)

    # -- path resolution ----------------------------------------------------

    def _resolve(self, base: Path, name: str) -> Path:
        candidate = Path(name)
        return candidate if candidate.is_absolute() else base / candidate

    def split_dir(self, split: str) -> Path:
        return self._resolve(self.root, self.split_dirs.get(split, split))

    def labels_dir(self, split: str) -> Path:
        return self._resolve(
            self.split_dir(split), self.labels_dirs.get(split, self.labels_dirname)
        )

    def images_dir(self, split: str) -> Path:
        return self._resolve(
            self.split_dir(split), self.images_dirs.get(split, self.images_dirname)
        )

    def available_splits(self) -> List[str]:
        """Splits whose labels directory actually exists on disk."""
        return [s for s in self.splits if self.labels_dir(s).is_dir()]

    def describe(self) -> Dict[str, Dict[str, str]]:
        """Resolved paths per split -- handy to print before a long run."""
        return {
            split: {
                "split_dir": str(self.split_dir(split)),
                "images_dir": str(self.images_dir(split)),
                "labels_dir": str(self.labels_dir(split)),
                "exists": str(self.labels_dir(split).is_dir()),
            }
            for split in self.splits
        }


@dataclass
class ConvertConfig:
    """Knobs for the conversion itself."""

    output_root: Optional[Path] = None
    """Where converted annotations go. ``None`` writes alongside the source
    labels (in-place), which fails on read-only mounts like /kaggle/input."""

    ann_dirname: str = "ann"
    """Name of the created annotation folder. The source ``labels/`` folder is
    never touched."""

    class_names: Optional[Sequence[str]] = None
    """Explicit class list. If ``None``, read from ``data_yaml``."""

    data_yaml: Optional[Path] = None
    """Path to data.yaml. If ``None``, looked up at ``<root>/data.yaml``."""

    clip_to_image: bool = True
    """Clamp vertices into [0, 1] before scaling to pixels."""

    coord_decimals: int = 1
    """Decimal places in output coordinates; 0 writes integers."""

    difficult: int = 0
    """Value written in the DOTA difficult column."""

    write_dota_header: bool = False
    """Prepend ``imagesource``/``gsd`` header lines. MMRotate tolerates both."""

    keep_empty: bool = True
    """Write an empty ``.txt`` for images with no objects, so negatives stay in
    the dataset instead of silently disappearing."""

    min_area_px: float = 0.0
    """Drop boxes smaller than this after scaling. 0 disables the check."""

    image_extensions: Sequence[str] = DEFAULT_IMAGE_EXTENSIONS

    default_image_size: Optional[Tuple[int, int]] = None
    """Fallback ``(width, height)`` when the image file is missing. Leave as
    ``None`` to skip such labels rather than guess."""

    overwrite: bool = True
    manifest_name: str = "conversion_manifest.json"
    include_file_records: bool = True
    """Store a per-file entry in the manifest (stem, size, object count).
    Useful later for board-wise splitting and tiling."""

    dry_run: bool = False

    def __post_init__(self) -> None:
        if self.output_root is not None:
            self.output_root = Path(self.output_root)
        if self.data_yaml is not None:
            self.data_yaml = Path(self.data_yaml)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class FileResult:
    stem: str
    label_path: str
    image_path: Optional[str]
    output_path: Optional[str]
    width: Optional[int]
    height: Optional[int]
    n_objects: int = 0
    n_dropped: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    skipped: bool = False

    def as_dict(self) -> Dict:
        return {
            "stem": self.stem,
            "image_path": self.image_path,
            "output_path": self.output_path,
            "width": self.width,
            "height": self.height,
            "n_objects": self.n_objects,
            "n_dropped": self.n_dropped,
            "class_counts": self.class_counts,
            "warnings": self.warnings,
            "skipped": self.skipped,
        }


@dataclass
class SplitResult:
    split: str
    labels_dir: str
    images_dir: str
    output_dir: str
    files: List[FileResult] = field(default_factory=list)

    @property
    def n_files(self) -> int:
        return sum(1 for f in self.files if not f.skipped)

    @property
    def n_skipped(self) -> int:
        return sum(1 for f in self.files if f.skipped)

    @property
    def n_objects(self) -> int:
        return sum(f.n_objects for f in self.files)

    @property
    def class_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for f in self.files:
            for name, n in f.class_counts.items():
                counts[name] = counts.get(name, 0) + n
        return dict(sorted(counts.items()))

    @property
    def warnings(self) -> List[str]:
        return [w for f in self.files for w in f.warnings]

    def image_size_stats(self) -> Dict[str, Optional[float]]:
        widths = sorted(f.width for f in self.files if f.width)
        heights = sorted(f.height for f in self.files if f.height)
        if not widths:
            return {"median_width": None, "median_height": None,
                    "min_width": None, "max_width": None,
                    "min_height": None, "max_height": None}
        mid = len(widths) // 2
        return {
            "median_width": float(widths[mid]),
            "median_height": float(heights[mid]),
            "min_width": float(widths[0]),
            "max_width": float(widths[-1]),
            "min_height": float(heights[0]),
            "max_height": float(heights[-1]),
        }

    def summary(self, include_files: bool = True) -> Dict:
        out: Dict = {
            "split": self.split,
            "labels_dir": self.labels_dir,
            "images_dir": self.images_dir,
            "output_dir": self.output_dir,
            "n_files_converted": self.n_files,
            "n_files_skipped": self.n_skipped,
            "n_objects": self.n_objects,
            "class_counts": self.class_counts,
            "image_size_stats": self.image_size_stats(),
            "warnings": self.warnings,
        }
        if include_files:
            out["files"] = [f.as_dict() for f in self.files]
        return out


# ---------------------------------------------------------------------------
# Conversion core
# ---------------------------------------------------------------------------

def convert_boxes_to_dota_lines(
    boxes: Iterable[OrientedBox],
    width: int,
    height: int,
    class_names: Sequence[str],
    cfg: ConvertConfig,
) -> Tuple[List[str], Dict[str, int], int]:
    """Normalised boxes -> DOTA text lines.

    Returns ``(lines, class_counts, n_dropped)``.
    """
    lines: List[str] = []
    counts: Dict[str, int] = {}
    dropped = 0

    for box in boxes:
        working = box.clipped(1.0, 1.0) if cfg.clip_to_image else box
        pixel_box = working.to_pixels(width, height)
        if cfg.difficult:
            pixel_box = OrientedBox(pixel_box.points, pixel_box.class_id, cfg.difficult)

        if cfg.min_area_px > 0 and pixel_box.area < cfg.min_area_px:
            dropped += 1
            continue

        name = resolve_class_name(class_names, pixel_box.class_id)
        lines.append(format_dota_line(pixel_box, name, cfg.coord_decimals))
        counts[sanitize_class_name(name)] = counts.get(sanitize_class_name(name), 0) + 1

    return lines, counts, dropped


def convert_label_file(
    label_path: Path,
    images_dir: Path,
    output_dir: Path,
    class_names: Sequence[str],
    cfg: ConvertConfig,
) -> FileResult:
    """Convert a single YOLO-OBB ``.txt`` and write the DOTA equivalent."""
    stem = label_path.stem
    result = FileResult(
        stem=stem,
        label_path=str(label_path),
        image_path=None,
        output_path=None,
        width=None,
        height=None,
    )

    boxes, parse_warnings = read_yolo_obb_file(label_path)
    result.warnings.extend(parse_warnings)

    image_path = find_image_for_stem(images_dir, stem, cfg.image_extensions)
    if image_path is not None:
        result.image_path = str(image_path)
        try:
            width, height = read_image_size(image_path)
        except Exception as exc:  # noqa: BLE001 - one bad file shouldn't kill the run
            result.warnings.append(f"{stem}: could not read image size ({exc})")
            width = height = None  # type: ignore[assignment]
    else:
        width = height = None  # type: ignore[assignment]
        result.warnings.append(f"{stem}: no matching image in {images_dir}")

    if width is None or height is None:
        if cfg.default_image_size is None:
            result.skipped = True
            return result
        width, height = cfg.default_image_size
        result.warnings.append(f"{stem}: using default_image_size {width}x{height}")

    result.width, result.height = int(width), int(height)

    if not boxes and not cfg.keep_empty:
        result.skipped = True
        return result

    lines, counts, dropped = convert_boxes_to_dota_lines(
        boxes, result.width, result.height, class_names, cfg
    )
    result.n_objects = len(lines)
    result.n_dropped = dropped
    result.class_counts = counts

    out_path = output_dir / f"{stem}.txt"
    result.output_path = str(out_path)

    if cfg.dry_run:
        return result
    if out_path.exists() and not cfg.overwrite:
        result.warnings.append(f"{stem}: output exists, not overwritten")
        return result

    header = ""
    if cfg.write_dota_header:
        header = "imagesource:PCB\ngsd:null\n"
    out_path.write_text(header + "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return result


def convert_split(
    layout: DatasetLayout,
    split: str,
    class_names: Sequence[str],
    cfg: ConvertConfig,
) -> SplitResult:
    """Convert every label file in one split."""
    labels_dir = layout.labels_dir(split)
    images_dir = layout.images_dir(split)

    if cfg.output_root is None:
        output_dir = layout.split_dir(split) / cfg.ann_dirname
    else:
        output_dir = cfg.output_root / split / cfg.ann_dirname

    if not labels_dir.is_dir():
        raise FileNotFoundError(f"labels directory not found for split '{split}': {labels_dir}")
    if not images_dir.is_dir():
        logger.warning("images directory not found for split '%s': %s", split, images_dir)

    if not cfg.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    result = SplitResult(
        split=split,
        labels_dir=str(labels_dir),
        images_dir=str(images_dir),
        output_dir=str(output_dir),
    )

    label_files = sorted(labels_dir.glob("*.txt"))
    logger.info("[%s] %d label files -> %s", split, len(label_files), output_dir)

    for label_path in label_files:
        result.files.append(
            convert_label_file(label_path, images_dir, output_dir, class_names, cfg)
        )

    logger.info(
        "[%s] converted %d files, %d objects, %d skipped",
        split, result.n_files, result.n_objects, result.n_skipped,
    )
    return result


def convert_dataset(
    layout: DatasetLayout,
    cfg: Optional[ConvertConfig] = None,
    splits: Optional[Sequence[str]] = None,
) -> Dict:
    """Convert a whole dataset and write a JSON manifest.

    Parameters
    ----------
    layout:
        Where the source dataset lives (see :class:`DatasetLayout`).
    cfg:
        Conversion options (see :class:`ConvertConfig`).
    splits:
        Restrict to these splits. Defaults to every split in ``layout``
        whose labels directory exists.

    Returns
    -------
    dict
        The manifest, also written to
        ``<output_root>/<manifest_name>`` unless ``dry_run``.
    """
    cfg = cfg or ConvertConfig()

    # -- class names --------------------------------------------------------
    if cfg.class_names is not None:
        class_names = list(cfg.class_names)
        names_source = "explicit"
    else:
        data_yaml = cfg.data_yaml or (layout.root / "data.yaml")
        class_names = load_class_names(Path(data_yaml))
        names_source = str(data_yaml)
    logger.info("class names (%s): %s", names_source, class_names)

    # -- splits -------------------------------------------------------------
    target_splits = list(splits) if splits is not None else layout.available_splits()
    missing = [s for s in (splits or layout.splits) if not layout.labels_dir(s).is_dir()]
    for split in missing:
        logger.warning("skipping split '%s': %s does not exist", split, layout.labels_dir(split))

    split_results = [convert_split(layout, s, class_names, cfg) for s in target_splits]

    # -- manifest -----------------------------------------------------------
    totals_class_counts: Dict[str, int] = {}
    for sr in split_results:
        for name, n in sr.class_counts.items():
            totals_class_counts[name] = totals_class_counts.get(name, 0) + n

    manifest = {
        "converter": {
            "module": "yolo_obb_to_dota",
            "version": __version__,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "source_format": "yolo-obb (normalised 4-point polygons)",
        "target_format": "dota (absolute-pixel 4-point polygons + class + difficult)",
        "layout": {
            "root": str(layout.root),
            "splits": list(layout.splits),
            "images_dirname": layout.images_dirname,
            "labels_dirname": layout.labels_dirname,
            "split_dirs": dict(layout.split_dirs),
            "images_dirs": dict(layout.images_dirs),
            "labels_dirs": dict(layout.labels_dirs),
            "resolved": layout.describe(),
        },
        "config": {
            "output_root": str(cfg.output_root) if cfg.output_root else None,
            "ann_dirname": cfg.ann_dirname,
            "clip_to_image": cfg.clip_to_image,
            "coord_decimals": cfg.coord_decimals,
            "difficult": cfg.difficult,
            "write_dota_header": cfg.write_dota_header,
            "keep_empty": cfg.keep_empty,
            "min_area_px": cfg.min_area_px,
            "default_image_size": list(cfg.default_image_size) if cfg.default_image_size else None,
            "dry_run": cfg.dry_run,
        },
        "class_names": list(class_names),
        "class_names_source": names_source,
        "totals": {
            "n_splits": len(split_results),
            "n_files_converted": sum(sr.n_files for sr in split_results),
            "n_files_skipped": sum(sr.n_skipped for sr in split_results),
            "n_objects": sum(sr.n_objects for sr in split_results),
            "class_counts": dict(sorted(totals_class_counts.items())),
            "n_warnings": sum(len(sr.warnings) for sr in split_results),
        },
        "splits": {
            sr.split: sr.summary(include_files=cfg.include_file_records)
            for sr in split_results
        },
    }

    if not cfg.dry_run:
        manifest_root = cfg.output_root or layout.root
        Path(manifest_root).mkdir(parents=True, exist_ok=True)
        manifest_path = Path(manifest_root) / cfg.manifest_name
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest["manifest_path"] = str(manifest_path)
        logger.info("manifest written to %s", manifest_path)

    return manifest


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_conversion(
    layout: DatasetLayout,
    cfg: ConvertConfig,
    split: str,
    tolerance_px: float = 1.0,
) -> Dict:
    """Round-trip check: DOTA back to normalised space, compared to the source.

    Reads each converted file, renormalises the coordinates using the image
    size, and compares against the original YOLO-OBB values. Reports the worst
    deviation in pixels and any count mismatches. Run this once after a
    conversion; it is cheap relative to training.
    """
    labels_dir = layout.labels_dir(split)
    images_dir = layout.images_dir(split)
    ann_dir = (
        layout.split_dir(split) / cfg.ann_dirname
        if cfg.output_root is None
        else cfg.output_root / split / cfg.ann_dirname
    )

    issues: List[str] = []
    max_dev = 0.0
    checked = 0

    for label_path in sorted(labels_dir.glob("*.txt")):
        stem = label_path.stem
        ann_path = ann_dir / f"{stem}.txt"
        if not ann_path.exists():
            issues.append(f"{stem}: no converted file")
            continue

        src_boxes, _ = read_yolo_obb_file(label_path)
        dst = read_dota_file(ann_path)

        if len(src_boxes) != len(dst):
            issues.append(f"{stem}: {len(src_boxes)} source boxes vs {len(dst)} converted")
            continue

        image_path = find_image_for_stem(images_dir, stem, cfg.image_extensions)
        if image_path is None:
            issues.append(f"{stem}: image missing, cannot verify")
            continue
        width, height = read_image_size(image_path)

        for src, (points, _name, _diff) in zip(src_boxes, dst):
            expected = src.clipped(1.0, 1.0) if cfg.clip_to_image else src
            expected = expected.to_pixels(width, height)
            for (ex, ey), (ax, ay) in zip(expected.points, points):
                max_dev = max(max_dev, abs(ex - ax), abs(ey - ay))
        checked += 1

    return {
        "split": split,
        "files_checked": checked,
        "max_deviation_px": round(max_dev, 4),
        "within_tolerance": max_dev <= tolerance_px,
        "issues": issues,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_kv(pairs: Optional[Sequence[str]]) -> Dict[str, str]:
    """Parse ``--labels-dirs train=annotations valid=labels`` style args."""
    out: Dict[str, str] = {}
    for item in pairs or []:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"expected key=value, got {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert YOLO-OBB labels to DOTA annotations without touching the source."
    )
    parser.add_argument("--root", required=True, help="dataset root containing the split folders")
    parser.add_argument("--output", default=None,
                        help="output root (required if the source is read-only, e.g. /kaggle/input)")
    parser.add_argument("--splits", nargs="*", default=["train", "valid", "test"])
    parser.add_argument("--images-dirname", default="images")
    parser.add_argument("--labels-dirname", default="labels")
    parser.add_argument("--ann-dirname", default="ann")
    parser.add_argument("--split-dirs", nargs="*", default=None,
                        help="per-split folder overrides, e.g. valid=validation")
    parser.add_argument("--labels-dirs", nargs="*", default=None,
                        help="per-split labels folder overrides, e.g. train=annotations")
    parser.add_argument("--images-dirs", nargs="*", default=None,
                        help="per-split images folder overrides")
    parser.add_argument("--data-yaml", default=None)
    parser.add_argument("--class-names", nargs="*", default=None)
    parser.add_argument("--decimals", type=int, default=1)
    parser.add_argument("--difficult", type=int, default=0)
    parser.add_argument("--min-area-px", type=float, default=0.0)
    parser.add_argument("--no-clip", action="store_true")
    parser.add_argument("--drop-empty", action="store_true",
                        help="do not write .txt files for images with no objects")
    parser.add_argument("--dota-header", action="store_true")
    parser.add_argument("--no-file-records", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true", help="round-trip check after converting")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    layout = DatasetLayout(
        root=Path(args.root),
        splits=args.splits,
        images_dirname=args.images_dirname,
        labels_dirname=args.labels_dirname,
        split_dirs=_parse_kv(args.split_dirs),
        images_dirs=_parse_kv(args.images_dirs),
        labels_dirs=_parse_kv(args.labels_dirs),
    )
    cfg = ConvertConfig(
        output_root=Path(args.output) if args.output else None,
        ann_dirname=args.ann_dirname,
        class_names=args.class_names,
        data_yaml=Path(args.data_yaml) if args.data_yaml else None,
        clip_to_image=not args.no_clip,
        coord_decimals=args.decimals,
        difficult=args.difficult,
        min_area_px=args.min_area_px,
        write_dota_header=args.dota_header,
        keep_empty=not args.drop_empty,
        include_file_records=not args.no_file_records,
        dry_run=args.dry_run,
    )

    manifest = convert_dataset(layout, cfg)
    totals = manifest["totals"]
    print(
        f"converted {totals['n_files_converted']} files, "
        f"{totals['n_objects']} objects, "
        f"{totals['n_files_skipped']} skipped, "
        f"{totals['n_warnings']} warnings"
    )
    print("class counts:", totals["class_counts"])

    if args.verify and not args.dry_run:
        for split in manifest["splits"]:
            report = verify_conversion(layout, cfg, split)
            print(
                f"[verify:{split}] {report['files_checked']} files, "
                f"max deviation {report['max_deviation_px']} px, "
                f"ok={report['within_tolerance']}, issues={len(report['issues'])}"
            )
            for issue in report["issues"][:10]:
                print("   ", issue)

    return 0


if __name__ == "__main__":
    sys.exit(main())
