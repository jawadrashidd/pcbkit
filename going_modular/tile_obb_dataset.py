"""
tile_obb_dataset.py
===================

Cut a **YOLO-OBB** dataset into tiles, placing every tile so that no annotated
object is ever sliced by a tile edge.

Why tile at all
---------------
PCB boards are large (~2600x2500) and defects are small (~50-150 px). Training
at ``imgsz=1024`` on whole boards downscales a defect to a handful of pixels.
Tiling keeps the source resolution: each crop is fed to the model at 1:1, so
the defect keeps its original size on screen.

How tiles are placed
--------------------
Two kinds of tile are produced.

**Anchor tiles** -- one per not-yet-covered object, positioned so that object
sits whole inside the crop and no *other* object is cut. Candidate offsets are
the centred position plus every offset that aligns a tile edge just outside a
neighbouring object's edge; that is exactly where a cutting window flips to a
clean one, so a few dozen candidates per axis cover the solution space. Any
other objects that land whole inside the chosen tile are marked covered too, so
a cluster costs one tile rather than one each.

**Grid tiles** -- a regular overlapping grid supplying background and context.
Any grid position that would cut an object is discarded; object-free ones are
subsampled at :attr:`TileConfig.bg_keep`.

The guarantee: every object appears complete in at least one tile, and no tile
anywhere contains a partial object. Nothing is labelled from a fragment, and no
visible fragment is left unlabelled.

When an object cannot be isolated at full tile size (a neighbour is too close),
the anchor falls back to a smaller crop -- 3/4, 1/2, 3/8, 1/4 of ``tile`` --
until it can. Those crops are letterboxed by YOLO at load time. The run reports
how many were shrunk.

Coordinate conventions
----------------------
INPUT and OUTPUT are both YOLO-OBB, one object per line::

    <class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>

    * 9 whitespace-separated tokens
    * 8 coordinates NORMALISED to [0, 1] w.r.t. the containing image
    * origin top-left, y increases downwards

Output coordinates are renormalised against the *tile*, not the source board.
Class ids pass through untouched, so the source ``data.yaml`` names still apply.

Layout
------
Source is the Roboflow export shape; output mirrors it::

    <root>/<split>/images/*.jpg      ->  <output_root>/<split>/images/*.jpg
    <root>/<split>/labels/*.txt      ->  <output_root>/<split>/labels/*.txt

Tile filenames encode provenance as ``<stem>__<x0>_<y0>_<w>x<h>.jpg``, so a
crop can always be traced back to its board and offset. The size suffix matters
because shrunk anchors and full tiles can share an origin.

A label file is written only when the tile contains at least one object, which
is the Ultralytics convention for background images.

Typical use
-----------
::

    from tile_obb_dataset import DatasetLayout, TileConfig, tile_dataset, verify_tiling

    layout = DatasetLayout(root="/kaggle/input/.../PCB-Defect.v4-50_trial_2.yolov8-obb")
    cfg = TileConfig(output_root="/kaggle/working/PCB-Defect-6", tile=1024, overlap=200)
    manifest = tile_dataset(layout, cfg)

    for split in manifest["splits"]:
        print(verify_tiling(layout, cfg, split))

Command line::

    python tile_obb_dataset.py --root <dataset_root> --output /kaggle/working/PCB-Defect-6 \\
        --tile 1024 --overlap 200 --verify
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from shapely.geometry import Polygon, box

__version__ = "0.2.0"

logger = logging.getLogger(__name__)

Bounds = Tuple[float, float, float, float]
Placement = Tuple[int, int, int, List[int]]

DEFAULT_IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
)

WHOLE = 0.999
"""Area fraction at or above which an object counts as fully inside a window."""

TOUCH = 1e-9
"""Area fraction below which an object counts as not present in a window at all."""


# ---------------------------------------------------------------------------
# Label IO
# ---------------------------------------------------------------------------

class LabelParseError(ValueError):
    """Raised for a malformed YOLO-OBB line."""


def read_yolo_obb_file(path: Path, width: int, height: int):
    """Read a YOLO-OBB ``.txt`` into pixel-space polygons.

    Returns ``(objects, warnings)`` where each object is
    ``(class_id, shapely.Polygon)`` in absolute pixels. Malformed lines are
    skipped with a warning rather than aborting the run -- a single bad line in
    one board should not lose the whole dataset.

    Self-intersecting corner orders (which some exports produce) are repaired
    with ``buffer(0)``; degenerate zero-area polygons are dropped.
    """
    objs: List[Tuple[int, Polygon]] = []
    warnings: List[str] = []
    if not path.exists():
        return objs, warnings

    text = path.read_text(encoding="utf-8", errors="replace")
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        vals = line.split()
        if len(vals) != 9:
            warnings.append(
                f"{path.name}:{lineno}: expected 9 tokens (class + 4 xy pairs), "
                f"got {len(vals)}"
            )
            continue
        try:
            pts = np.array(vals[1:], float).reshape(4, 2) * [width, height]
            class_id = int(float(vals[0]))
        except ValueError:
            warnings.append(f"{path.name}:{lineno}: non-numeric token")
            continue
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.area > 0:
            objs.append((class_id, poly))
    return objs, warnings


def obb_line(class_id: int, poly: Polygon, x0: int, y0: int, tw: int, th: int) -> str:
    """Format one polygon as a YOLO-OBB line normalised against the tile."""
    pts = np.array(poly.exterior.coords[:-1], np.float32)
    if len(pts) != 4:  # repaired polygon: refit a rectangle
        pts = cv2.boxPoints(cv2.minAreaRect(pts))
    pts = pts - [x0, y0]
    pts[:, 0] = np.clip(pts[:, 0], 0, tw) / tw
    pts[:, 1] = np.clip(pts[:, 1], 0, th) / th
    return f"{class_id} " + " ".join(f"{v:.6f}" for v in pts.reshape(-1))


def tile_name(stem: str, x0: int, y0: int, tw: int, th: int) -> str:
    """Provenance-encoding tile stem: source board, offset, and crop size."""
    return f"{stem}__{x0}_{y0}_{tw}x{th}"


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def window(x0: int, y0: int, width: int, height: int, tile: int):
    """Clip a tile origin to the image, returning ``(x0, y0, x1, y1)``."""
    return x0, y0, min(x0 + tile, width), min(y0 + tile, height)


def classify(win, objs) -> Tuple[List[int], bool]:
    """How a window sees the objects: ``(indices fully inside, any cut?)``.

    Returns early on the first cut object -- callers only ever need to know
    *whether* the window is clean, never which object spoiled it.
    """
    inside: List[int] = []
    for k, (_, poly) in enumerate(objs):
        frac = poly.intersection(win).area / poly.area
        if frac < TOUCH:
            continue
        if frac >= WHOLE:
            inside.append(k)
        else:
            return inside, True
    return inside, False


def axis_candidates(lo, hi, tile, length, edges) -> List[int]:
    """Start offsets on one axis whose ``[start, start+tile]`` span holds ``[lo, hi]``.

    The candidate set is the centred position plus, for every object edge, the
    two offsets that put a tile boundary just outside it. Those are precisely
    the positions where a cutting window becomes a clean one, so searching them
    is equivalent to searching every pixel offset -- at a few dozen candidates
    instead of a thousand.
    """
    if hi - lo > tile:
        return []
    want = (lo + hi) / 2 - tile / 2
    raw = {0, max(length - tile, 0), want, lo, hi - tile}
    for e in edges:
        raw.add(e + 1)          # tile starts just past an edge (object pushed out left)
        raw.add(e - 1 - tile)   # tile ends just before an edge (pushed out right)
    out = set()
    for c in raw:
        c = int(np.floor(min(max(c, 0), max(length - tile, 0))))
        if c <= lo and c + tile >= hi:
            out.add(c)
    return sorted(out, key=lambda c: abs(c - want))


def try_size(bnd: Bounds, objs, edges_x, edges_y, width, height, tile):
    """Find a tile of exactly this size holding the object whole and cutting none."""
    a, c, b, d = bnd
    xs = axis_candidates(a, min(b, width), tile, width, edges_x)
    ys = axis_candidates(c, min(d, height), tile, height, edges_y)
    if not xs or not ys:
        return None
    cx, cy = xs[0], ys[0]
    combos = sorted(((x, y) for x in xs for y in ys),
                    key=lambda p: abs(p[0] - cx) + abs(p[1] - cy))
    for x0, y0 in combos:
        x0, y0, x1, y1 = window(x0, y0, width, height, tile)
        inside, cut = classify(box(x0, y0, x1, y1), objs)
        if not cut:
            return (x0, y0, tile), inside
    return None


def place_anchor(bnd, objs, edges_x, edges_y, width, height, tile, shrink_steps):
    """Isolate one object, shrinking the crop if full size always cuts a neighbour."""
    need = max(bnd[2] - bnd[0], bnd[3] - bnd[1])
    for frac in shrink_steps:
        size = int(tile * frac)
        if size < need:
            break
        got = try_size(bnd, objs, edges_x, edges_y, width, height, size)
        if got is not None:
            return got
    return None


def grid_starts(length: int, tile: int, step: int) -> List[int]:
    """Regular start offsets along one axis; the last is flush with the image edge."""
    if length <= tile:
        return [0]
    return sorted(set(list(range(0, length - tile, step)) + [length - tile]))


# ---------------------------------------------------------------------------
# Layout + configuration
# ---------------------------------------------------------------------------

@dataclass
class DatasetLayout:
    """Where the source dataset lives and what its folders are called.

    Every name can be overridden globally or per split. A per-split override may
    be a plain folder name (resolved relative to that split's directory) or an
    absolute path, which covers datasets whose splits sit in different roots.

    Examples
    --------
    Default Roboflow shape::

        DatasetLayout(root="/kaggle/input/.../PCB-Defect.v4-50_trial_2.yolov8-obb")

    A set with no validation split::

        DatasetLayout(root=..., splits=("train", "test"))
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

    def _resolve(self, base: Path, name: str) -> Path:
        candidate = Path(name)
        return candidate if candidate.is_absolute() else base / candidate

    def split_dir(self, split: str) -> Path:
        return self._resolve(self.root, self.split_dirs.get(split, split))

    def images_dir(self, split: str) -> Path:
        return self._resolve(
            self.split_dir(split), self.images_dirs.get(split, self.images_dirname)
        )

    def labels_dir(self, split: str) -> Path:
        return self._resolve(
            self.split_dir(split), self.labels_dirs.get(split, self.labels_dirname)
        )

    def available_splits(self) -> List[str]:
        """Splits whose images directory actually exists on disk."""
        return [s for s in self.splits if self.images_dir(s).is_dir()]

    def describe(self) -> Dict[str, Dict[str, str]]:
        """Resolved paths per split -- handy to print before a long run."""
        return {
            split: {
                "split_dir": str(self.split_dir(split)),
                "images_dir": str(self.images_dir(split)),
                "labels_dir": str(self.labels_dir(split)),
                "exists": str(self.images_dir(split).is_dir()),
            }
            for split in self.splits
        }


@dataclass
class TileConfig:
    """Knobs for the tiling itself."""

    output_root: Path = Path("PCB-Defect-tiles")
    """Where tiles are written, as ``<output_root>/<split>/{images,labels}/``.

    Unlike an in-place converter this always needs a destination -- tiling
    produces new images, not sidecar files. Point it at ``/kaggle/working``;
    ``/kaggle/input`` is read-only.
    """

    tile: int = 1024
    """Crop size in source pixels. Train with ``imgsz=tile`` so nothing is
    rescaled. Objects larger than this cannot be enclosed and are reported."""

    overlap: int = 200
    """Overlap between *grid* tiles only. It no longer guards against sliced
    objects -- anchor placement does that at any value -- so treat it as a
    background sampling-density knob. Raise it if many grid tiles are dropped."""

    bg_keep: float = 0.3
    """Fraction of object-free grid tiles to keep. 1.0 keeps every clean grid
    position; low values stop the set being swamped by blank board."""

    shrink_steps: Sequence[float] = (1.0, 0.75, 0.5, 0.375, 0.25)
    """Crop-size fractions tried when a full-size tile always cuts a neighbour.
    Set to ``(1.0,)`` to disable shrinking and drop such objects instead."""

    seed: int = 0
    """Seeds the background subsampling, so runs are reproducible."""

    jpeg_quality: int = 98
    """Encoder quality for written tiles. High, because artefacts at this scale
    are the same size as the defects."""

    image_extensions: Sequence[str] = DEFAULT_IMAGE_EXTENSIONS

    write_data_yaml: bool = True
    """Copy ``names`` from the source ``data.yaml`` into one for the tiled set."""

    data_yaml: Optional[Path] = None
    """Source data.yaml. If ``None``, looked up at ``<root>/data.yaml``."""

    val_split: str = "test"
    """Which split the generated data.yaml points ``val`` at."""

    manifest_name: str = "tiling_manifest.json"
    include_file_records: bool = True
    """Store a per-board entry in the manifest (size, tiles produced, warnings)."""

    dry_run: bool = False
    """Plan and count everything without writing any image or label."""

    def __post_init__(self) -> None:
        self.output_root = Path(self.output_root)
        if self.data_yaml is not None:
            self.data_yaml = Path(self.data_yaml)
        if not 0 <= self.overlap < self.tile:
            raise ValueError(f"overlap must be in [0, tile); got {self.overlap} with tile={self.tile}")
        if not 0.0 <= self.bg_keep <= 1.0:
            raise ValueError(f"bg_keep must be in [0, 1]; got {self.bg_keep}")

    @property
    def step(self) -> int:
        """Distance between consecutive grid tile origins."""
        return self.tile - self.overlap

    def images_dir(self, split: str) -> Path:
        return self.output_root / split / "images"

    def labels_dir(self, split: str) -> Path:
        return self.output_root / split / "labels"


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class ImageResult:
    """What tiling one source board produced."""

    stem: str
    image_path: str
    width: Optional[int] = None
    height: Optional[int] = None
    n_objects: int = 0
    n_tiles: int = 0
    n_anchor: int = 0
    n_background: int = 0
    n_instances: int = 0
    n_shrunk: int = 0
    n_grid_dropped: int = 0
    n_oversize: int = 0
    n_unplaceable: int = 0
    warnings: List[str] = field(default_factory=list)
    skipped: bool = False

    def as_dict(self) -> Dict:
        return {
            "stem": self.stem,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "n_objects": self.n_objects,
            "n_tiles": self.n_tiles,
            "n_anchor": self.n_anchor,
            "n_background": self.n_background,
            "n_instances": self.n_instances,
            "n_shrunk": self.n_shrunk,
            "n_grid_dropped": self.n_grid_dropped,
            "n_oversize": self.n_oversize,
            "n_unplaceable": self.n_unplaceable,
            "warnings": self.warnings,
            "skipped": self.skipped,
        }


@dataclass
class SplitResult:
    """Aggregate over every board in one split."""

    split: str
    images_dir: str
    labels_dir: str
    output_images_dir: str
    output_labels_dir: str
    images: List[ImageResult] = field(default_factory=list)

    def _sum(self, attr: str) -> int:
        return sum(getattr(i, attr) for i in self.images)

    @property
    def n_images(self) -> int:
        return sum(1 for i in self.images if not i.skipped)

    @property
    def n_skipped(self) -> int:
        return sum(1 for i in self.images if i.skipped)

    @property
    def n_tiles(self) -> int:
        return self._sum("n_tiles")

    @property
    def n_objects(self) -> int:
        return self._sum("n_objects")

    @property
    def n_instances(self) -> int:
        return self._sum("n_instances")

    @property
    def warnings(self) -> List[str]:
        return [w for i in self.images for w in i.warnings]

    @property
    def tiles_per_image(self) -> Optional[float]:
        return round(self.n_tiles / self.n_images, 2) if self.n_images else None

    def summary(self, include_files: bool = True) -> Dict:
        out: Dict = {
            "split": self.split,
            "images_dir": self.images_dir,
            "labels_dir": self.labels_dir,
            "output_images_dir": self.output_images_dir,
            "output_labels_dir": self.output_labels_dir,
            "n_images": self.n_images,
            "n_images_skipped": self.n_skipped,
            "n_tiles": self.n_tiles,
            "n_anchor_tiles": self._sum("n_anchor"),
            "n_background_tiles": self._sum("n_background"),
            "n_objects": self.n_objects,
            "n_labeled_instances": self.n_instances,
            "n_shrunk_anchors": self._sum("n_shrunk"),
            "n_grid_dropped": self._sum("n_grid_dropped"),
            "n_oversize_objects": self._sum("n_oversize"),
            "n_unplaceable_objects": self._sum("n_unplaceable"),
            "tiles_per_image": self.tiles_per_image,
            "warnings": self.warnings,
        }
        if include_files:
            out["images"] = [i.as_dict() for i in self.images]
        return out

    def issues(self) -> List[str]:
        """Human-readable problems worth acting on, or an empty list."""
        out = []
        if self._sum("n_oversize"):
            out.append(
                f"{self._sum('n_oversize')} objects are larger than the tile size "
                f"and were dropped; increase tile"
            )
        if self._sum("n_unplaceable"):
            out.append(
                f"{self._sum('n_unplaceable')} objects sit too close to a neighbour "
                f"to isolate even at the smallest shrink step; increase tile"
            )
        if self.warnings:
            out.append(f"{len(self.warnings)} malformed label lines; export in YOLO-OBB format")
        return out


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def plan_tiles(objs, width, height, cfg: TileConfig, rng, result: ImageResult) -> List[Placement]:
    """Choose every tile position for one board.

    Returns ``[(x0, y0, size, [indices of objects fully inside])]``. Anchors come
    first, then surviving grid positions. Counters land on ``result``.
    """
    bounds = [poly.bounds for _, poly in objs]
    edges_x = sorted({v for bd in bounds for v in (bd[0], bd[2])})
    edges_y = sorted({v for bd in bounds for v in (bd[1], bd[3])})

    plan: List[Placement] = []
    taken = set()
    covered = [False] * len(objs)

    # 1. anchors -- driven by the object list, so no object can be missed
    for k, bnd in enumerate(bounds):
        if covered[k]:
            continue
        if bnd[2] - bnd[0] > cfg.tile or bnd[3] - bnd[1] > cfg.tile:
            result.n_oversize += 1
            covered[k] = True  # nothing this tile size can do
            continue
        found = place_anchor(
            bnd, objs, edges_x, edges_y, width, height, cfg.tile, cfg.shrink_steps
        )
        if found is None:
            result.n_unplaceable += 1
            continue
        (x0, y0, size), inside = found
        for i in inside:
            covered[i] = True
        if (x0, y0, size) in taken:
            continue
        taken.add((x0, y0, size))
        plan.append((x0, y0, size, inside))
        result.n_anchor += 1
        result.n_shrunk += size < cfg.tile

    # 2. background / context grid -- discard anything that would cut an object
    for y0 in grid_starts(height, cfg.tile, cfg.step):
        for x0 in grid_starts(width, cfg.tile, cfg.step):
            if (x0, y0, cfg.tile) in taken:
                continue
            gx0, gy0, gx1, gy1 = window(x0, y0, width, height, cfg.tile)
            inside, cut = classify(box(gx0, gy0, gx1, gy1), objs)
            if cut:
                result.n_grid_dropped += 1
                continue
            if not inside and rng.random() > cfg.bg_keep:
                continue
            taken.add((x0, y0, cfg.tile))
            plan.append((x0, y0, cfg.tile, inside))
            result.n_background += not inside

    return plan


# ---------------------------------------------------------------------------
# Tiling core
# ---------------------------------------------------------------------------

def tile_image(
    image_path: Path,
    labels_dir: Path,
    out_images: Path,
    out_labels: Path,
    cfg: TileConfig,
    rng: random.Random,
) -> ImageResult:
    """Tile one source board and write its crops and labels."""
    result = ImageResult(stem=image_path.stem, image_path=str(image_path))

    im = cv2.imread(str(image_path))
    if im is None:
        result.warnings.append(f"{image_path.name}: could not decode image")
        result.skipped = True
        return result

    height, width = im.shape[:2]
    result.width, result.height = width, height

    objs, warnings = read_yolo_obb_file(labels_dir / f"{image_path.stem}.txt", width, height)
    result.warnings.extend(warnings)
    result.n_objects = len(objs)

    for x0, y0, size, inside in plan_tiles(objs, width, height, cfg, rng, result):
        x0, y0, x1, y1 = window(x0, y0, width, height, size)
        tw, th = x1 - x0, y1 - y0
        lines = [obb_line(objs[k][0], objs[k][1], x0, y0, tw, th) for k in inside]
        name = tile_name(image_path.stem, x0, y0, tw, th)

        if not cfg.dry_run:
            cv2.imwrite(
                str(out_images / f"{name}.jpg"),
                im[y0:y1, x0:x1],
                [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality],
            )
            if lines:
                (out_labels / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

        result.n_instances += len(lines)
        result.n_tiles += 1

    return result


def tile_split(layout: DatasetLayout, split: str, cfg: TileConfig, rng: random.Random) -> SplitResult:
    """Tile every board in one split."""
    images_dir = layout.images_dir(split)
    labels_dir = layout.labels_dir(split)
    out_images = cfg.images_dir(split)
    out_labels = cfg.labels_dir(split)

    if not images_dir.is_dir():
        raise FileNotFoundError(f"images directory not found for split '{split}': {images_dir}")
    if not labels_dir.is_dir():
        logger.warning("labels directory not found for split '%s': %s", split, labels_dir)

    if not cfg.dry_run:
        try:
            out_images.mkdir(parents=True, exist_ok=True)
            out_labels.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OSError(
                f"cannot create {out_images} ({exc}). Point TileConfig(output_root=...) "
                f"at a writable location such as /kaggle/working."
            ) from exc

    result = SplitResult(
        split=split,
        images_dir=str(images_dir),
        labels_dir=str(labels_dir),
        output_images_dir=str(out_images),
        output_labels_dir=str(out_labels),
    )

    lowered = {e.lower() for e in cfg.image_extensions}
    image_files = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in lowered
    )
    logger.info("[%s] %d source images -> %s", split, len(image_files), out_images)

    for image_path in image_files:
        result.images.append(
            tile_image(image_path, labels_dir, out_images, out_labels, cfg, rng)
        )

    logger.info(
        "[%s] %d images -> %d tiles (%d anchored, %d background), %d objects -> %d instances",
        split, result.n_images, result.n_tiles,
        result._sum("n_anchor"), result._sum("n_background"),
        result.n_objects, result.n_instances,
    )
    return result


def write_data_yaml(layout: DatasetLayout, cfg: TileConfig, splits: Sequence[str]) -> Optional[Path]:
    """Write a ``data.yaml`` for the tiled set, reusing the source class names.

    Class ids pass through tiling untouched, so the source ``names`` mapping is
    still correct. Returns the written path, or ``None`` if the source yaml is
    missing or disabled.
    """
    import yaml  # noqa: WPS433 - keep pyyaml optional

    src_yaml = cfg.data_yaml or (layout.root / "data.yaml")
    if not Path(src_yaml).exists():
        logger.warning("no data.yaml at %s; skipping tiled data.yaml", src_yaml)
        return None

    with Path(src_yaml).open("r", encoding="utf-8") as fh:
        names = (yaml.safe_load(fh) or {}).get("names")
    if names is None:
        logger.warning("no 'names' key in %s; skipping tiled data.yaml", src_yaml)
        return None

    val = cfg.val_split if cfg.val_split in splits else splits[-1]
    doc = {
        "path": str(cfg.output_root),
        "train": "train/images" if "train" in splits else f"{splits[0]}/images",
        "val": f"{val}/images",
        "names": names,
    }
    out = cfg.output_root / "data.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False)
    logger.info("data.yaml written to %s", out)
    return out


def tile_dataset(
    layout: DatasetLayout,
    cfg: Optional[TileConfig] = None,
    splits: Optional[Sequence[str]] = None,
) -> Dict:
    """Tile a whole dataset and write a JSON manifest.

    Parameters
    ----------
    layout:
        Where the source dataset lives (see :class:`DatasetLayout`).
    cfg:
        Tiling options (see :class:`TileConfig`).
    splits:
        Restrict to these splits. Defaults to every split in ``layout`` whose
        images directory exists.

    Returns
    -------
    dict
        The manifest, also written to ``<output_root>/<manifest_name>`` unless
        ``dry_run``.
    """
    cfg = cfg or TileConfig()
    rng = random.Random(cfg.seed)

    target_splits = list(splits) if splits is not None else layout.available_splits()
    missing = [s for s in (splits or layout.splits) if not layout.images_dir(s).is_dir()]
    for split in missing:
        logger.warning("skipping split '%s': %s does not exist", split, layout.images_dir(split))
    if not target_splits:
        raise FileNotFoundError(f"no usable splits under {layout.root}")

    split_results = [tile_split(layout, s, cfg, rng) for s in target_splits]

    data_yaml_path = None
    if cfg.write_data_yaml and not cfg.dry_run:
        written = write_data_yaml(layout, cfg, target_splits)
        data_yaml_path = str(written) if written else None

    manifest = {
        "tiler": {
            "module": "tile_obb_dataset",
            "version": __version__,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "source_format": "yolo-obb (normalised 4-point polygons)",
        "target_format": "yolo-obb tiles (coordinates renormalised against each tile)",
        "guarantee": (
            "every object appears whole in at least one tile; no tile contains a "
            "partial object"
        ),
        "layout": {
            "root": str(layout.root),
            "splits": list(layout.splits),
            "images_dirname": layout.images_dirname,
            "labels_dirname": layout.labels_dirname,
            "resolved": layout.describe(),
        },
        "config": {
            "output_root": str(cfg.output_root),
            "tile": cfg.tile,
            "overlap": cfg.overlap,
            "step": cfg.step,
            "bg_keep": cfg.bg_keep,
            "shrink_steps": list(cfg.shrink_steps),
            "seed": cfg.seed,
            "jpeg_quality": cfg.jpeg_quality,
            "dry_run": cfg.dry_run,
        },
        "data_yaml": data_yaml_path,
        "totals": {
            "n_splits": len(split_results),
            "n_images": sum(sr.n_images for sr in split_results),
            "n_images_skipped": sum(sr.n_skipped for sr in split_results),
            "n_tiles": sum(sr.n_tiles for sr in split_results),
            "n_anchor_tiles": sum(sr._sum("n_anchor") for sr in split_results),
            "n_background_tiles": sum(sr._sum("n_background") for sr in split_results),
            "n_objects": sum(sr.n_objects for sr in split_results),
            "n_labeled_instances": sum(sr.n_instances for sr in split_results),
            "n_shrunk_anchors": sum(sr._sum("n_shrunk") for sr in split_results),
            "n_grid_dropped": sum(sr._sum("n_grid_dropped") for sr in split_results),
            "n_oversize_objects": sum(sr._sum("n_oversize") for sr in split_results),
            "n_unplaceable_objects": sum(sr._sum("n_unplaceable") for sr in split_results),
            "n_warnings": sum(len(sr.warnings) for sr in split_results),
        },
        "splits": {
            sr.split: sr.summary(include_files=cfg.include_file_records)
            for sr in split_results
        },
        "issues": {sr.split: sr.issues() for sr in split_results if sr.issues()},
    }

    if not cfg.dry_run:
        cfg.output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = cfg.output_root / cfg.manifest_name
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest["manifest_path"] = str(manifest_path)
        logger.info("manifest written to %s", manifest_path)

    return manifest


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_tiling(
    layout: DatasetLayout,
    cfg: TileConfig,
    split: str,
    iou_tolerance: float = 0.97,
) -> Dict:
    """Independently re-check the two guarantees against what is on disk.

    For every written tile this reparses the *source* labels, recomputes which
    objects the crop window sees, and asserts that none is partial and that the
    written label count matches. Each written box is also reprojected back to
    board coordinates and compared to the original polygon by IoU, which catches
    any normalisation or offset error.

    Finally it checks coverage: every source object must appear whole in at
    least one tile. Cheap relative to training -- run it once after tiling.
    """
    images_dir = layout.images_dir(split)
    labels_dir = layout.labels_dir(split)
    out_images = cfg.images_dir(split)
    out_labels = cfg.labels_dir(split)

    issues: List[str] = []
    lowered = {e.lower() for e in cfg.image_extensions}

    source: Dict[str, list] = {}
    coverage: Dict[str, List[bool]] = {}
    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() not in lowered:
            continue
        im = cv2.imread(str(p))
        if im is None:
            continue
        h, w = im.shape[:2]
        objs, _ = read_yolo_obb_file(labels_dir / f"{p.stem}.txt", w, h)
        source[p.stem] = objs
        coverage[p.stem] = [False] * len(objs)

    n_tiles = 0
    worst_iou = 1.0
    for tile_path in sorted(out_images.iterdir()):
        if tile_path.suffix.lower() not in lowered:
            continue
        n_tiles += 1
        try:
            stem, _, suffix = tile_path.stem.rpartition("__")
            x0_s, y0_s, _size = suffix.split("_")
            x0, y0 = int(x0_s), int(y0_s)
        except ValueError:
            issues.append(f"{tile_path.name}: unparseable tile filename")
            continue
        if stem not in source:
            issues.append(f"{tile_path.name}: no source board named {stem!r}")
            continue

        tim = cv2.imread(str(tile_path))
        if tim is None:
            issues.append(f"{tile_path.name}: could not decode tile")
            continue
        th, tw = tim.shape[:2]

        objs = source[stem]
        inside, cut = classify(box(x0, y0, x0 + tw, y0 + th), objs)
        if cut:
            issues.append(f"{tile_path.name}: contains a partial object")
            continue
        for k in inside:
            coverage[stem][k] = True

        label_path = out_labels / f"{tile_path.stem}.txt"
        written = label_path.read_text(encoding="utf-8").splitlines() if label_path.exists() else []
        if len(written) != len(inside):
            issues.append(
                f"{tile_path.name}: {len(written)} labels written but {len(inside)} objects inside"
            )
            continue

        for line, k in zip(written, inside):
            pts = np.array(line.split()[1:], float).reshape(4, 2) * [tw, th] + [x0, y0]
            back = Polygon(pts)
            if not back.is_valid:
                back = back.buffer(0)
            union = back.union(objs[k][1]).area
            iou = back.intersection(objs[k][1]).area / union if union else 0.0
            worst_iou = min(worst_iou, iou)
            if iou < iou_tolerance:
                issues.append(f"{tile_path.name}: box geometry drifted (IoU {iou:.3f})")

    n_objects = sum(len(v) for v in coverage.values())
    n_covered = sum(sum(v) for v in coverage.values())
    for stem, flags in coverage.items():
        if not all(flags):
            issues.append(f"{stem}: {flags.count(False)} objects never appear whole in any tile")

    return {
        "split": split,
        "output_images_dir": str(out_images),
        "tiles_checked": n_tiles,
        "objects": n_objects,
        "objects_covered": n_covered,
        "worst_box_iou": round(worst_iou, 4) if n_tiles else None,
        "ok": not issues and n_covered == n_objects,
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
        description="Tile a YOLO-OBB dataset without ever slicing an annotated object.",
        epilog=(
            "Tiles go to <output>/<split>/{images,labels}/. --overlap affects only the "
            "background grid; object coverage is guaranteed by anchor placement at any "
            "value. Raise --bg-keep for more blank-board tiles, lower it to keep the "
            "positive ratio high."
        ),
    )
    parser.add_argument("--root", required=True, help="dataset root containing the split folders")
    parser.add_argument("--output", required=True, help="where tiles are written (must be writable)")
    parser.add_argument("--splits", nargs="*", default=["train", "valid", "test"])
    parser.add_argument("--images-dirname", default="images")
    parser.add_argument("--labels-dirname", default="labels")
    parser.add_argument("--split-dirs", nargs="*", default=None,
                        help="per-split folder overrides, e.g. valid=validation")
    parser.add_argument("--images-dirs", nargs="*", default=None)
    parser.add_argument("--labels-dirs", nargs="*", default=None)
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--bg-keep", type=float, default=0.3)
    parser.add_argument("--no-shrink", action="store_true",
                        help="never shrink an anchor below --tile; drop uncoverable objects")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jpeg-quality", type=int, default=98)
    parser.add_argument("--data-yaml", default=None)
    parser.add_argument("--val-split", default="test")
    parser.add_argument("--no-data-yaml", action="store_true")
    parser.add_argument("--no-file-records", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true", help="re-check the guarantees afterwards")
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
    cfg = TileConfig(
        output_root=Path(args.output),
        tile=args.tile,
        overlap=args.overlap,
        bg_keep=args.bg_keep,
        shrink_steps=(1.0,) if args.no_shrink else TileConfig.shrink_steps,
        seed=args.seed,
        jpeg_quality=args.jpeg_quality,
        data_yaml=Path(args.data_yaml) if args.data_yaml else None,
        val_split=args.val_split,
        write_data_yaml=not args.no_data_yaml,
        include_file_records=not args.no_file_records,
        dry_run=args.dry_run,
    )

    manifest = tile_dataset(layout, cfg)
    totals = manifest["totals"]
    print(
        f"{totals['n_images']} images -> {totals['n_tiles']} tiles "
        f"({totals['n_anchor_tiles']} anchored, {totals['n_background_tiles']} background), "
        f"{totals['n_objects']} objects -> {totals['n_labeled_instances']} labeled instances"
    )
    for split, summary in manifest["splits"].items():
        print(f"  {split} -> {summary['output_images_dir']} "
              f"({summary['n_tiles']} tiles, {summary['tiles_per_image']} per image)")
    print(f"  {totals['n_grid_dropped']} grid tiles dropped for cutting an object, "
          f"{totals['n_shrunk_anchors']} anchors shrunk")
    for split, issues in manifest.get("issues", {}).items():
        for issue in issues:
            print(f"  WARNING [{split}] {issue}")

    if args.verify and not args.dry_run:
        for split in manifest["splits"]:
            report = verify_tiling(layout, cfg, split)
            print(
                f"[verify:{split}] {report['tiles_checked']} tiles, "
                f"{report['objects_covered']}/{report['objects']} objects covered, "
                f"worst box IoU {report['worst_box_iou']}, ok={report['ok']}"
            )
            for issue in report["issues"][:10]:
                print("   ", issue)

    return 0


if __name__ == "__main__":
    sys.exit(main())
