"""
obb_to_coco.py
==============

Convert **YOLO-OBB** label files into **COCO** instance-detection JSON, one file
per split, written beside that split's images. The source ``labels/`` folder is
never touched.

Why this exists
---------------
Horizontal-box detectors (MMDetection, Detectron2, torchvision) read COCO.
Oriented-box detectors (Ultralytics OBB, MMRotate) read YOLO-OBB or DOTA. To
compare the two families fairly they must train on *identical pixels*, so the
right pipeline keeps orientation as long as possible and flattens to
horizontal boxes only at the very end::

    YOLO-OBB export
        |  tile_obb_dataset.py        anchor placement on the true polygons
        v
    <tiles>/<split>/{images,labels}   YOLO-OBB   -> oriented detectors
        |  obb_to_coco.py              min enclosing rect, per tile
        v
    <tiles>/<split>/_annotations.coco.json       -> horizontal detectors

Tiling from a COCO export instead would place tiles against inflated
horizontal boxes and the two arms would see different crops.

Coordinate conventions
----------------------
INPUT -- YOLO-OBB, one object per line::

    <class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>

    * 9 whitespace-separated tokens
    * class_id indexes ``names`` in data.yaml (0-based)
    * 8 coordinates NORMALISED to [0, 1] w.r.t. image width / height

OUTPUT -- COCO detection JSON::

    images[]       id, file_name, width, height  (+ tile provenance, optional)
    annotations[]  id, image_id, category_id, bbox [x, y, w, h] in PIXELS,
                   area, iscrowd, segmentation (the 4-point polygon, optional)
    categories[]   id, name, supercategory

``bbox`` is the axis-aligned minimum enclosing rectangle of the rotated
polygon. ``segmentation`` optionally carries the original polygon so the COCO
file remains lossless -- anything reading only ``bbox`` sees a horizontal box,
anything reading ``segmentation`` can recover the orientation.

``category_id`` is 1-based by default (COCO convention; many loaders treat 0
as background). Roboflow additionally prepends a dummy supercategory at id 0;
``ConvertConfig(roboflow_supercategory="...")`` reproduces that shape if you
need byte-compatibility with a Roboflow export.

Layout
------
Default source shape (Roboflow / tiler output)::

    <root>/
        data.yaml
        train/  images/  labels/
        valid/  images/  labels/
        test/   images/  labels/

Default output: ``<root>/<split>/_annotations.coco.json`` -- the JSON sits
beside the split's ``images/`` and ``labels/`` folders. Set
``ConvertConfig(output_root=...)`` to write the JSONs under a different root
instead, mirroring the split names.

Every folder name is overridable globally or per split via
:class:`DatasetLayout`; every output choice via :class:`ConvertConfig`.

Typical use
-----------
::

    from obb_to_coco import DatasetLayout, ConvertConfig, convert_dataset, verify_conversion

    layout = DatasetLayout(root="/kaggle/working/PCB-Defect-tiles")
    cfg = ConvertConfig()                      # in-place, beside images/
    manifest = convert_dataset(layout, cfg)

    for split in manifest["splits"]:
        print(verify_conversion(layout, cfg, split))

Then in an MMDetection config::

    data_root = '/kaggle/working/PCB-Defect-tiles/'
    train_dataloader = dict(dataset=dict(
        data_root=data_root,
        ann_file='train/_annotations.coco.json',
        data_prefix=dict(img='train/images/')))

Command line::

    python obb_to_coco.py --root /kaggle/working/PCB-Defect-tiles --verify
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

__version__ = "0.1.0"

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
)

# Matches the tiler's provenance-encoding stem: <board>__<x0>_<y0>_<w>x<h>
TILE_STEM_RE = re.compile(r"^(?P<board>.+)__(?P<x0>\d+)_(?P<y0>\d+)_(?P<w>\d+)x(?P<h>\d+)$")


# ---------------------------------------------------------------------------
# Label IO
# ---------------------------------------------------------------------------

class LabelParseError(ValueError):
    """Raised for a malformed YOLO-OBB line."""


def parse_yolo_obb_line(line: str) -> Tuple[int, List[float]]:
    """One YOLO-OBB line -> ``(class_id, [x1,y1,...,x4,y4])`` in normalised space."""
    tokens = line.split()
    if len(tokens) != 9:
        raise LabelParseError(
            f"expected 9 tokens (class + 4 xy pairs), got {len(tokens)}: {line.strip()!r}"
        )
    try:
        return int(float(tokens[0])), [float(t) for t in tokens[1:]]
    except ValueError as exc:
        raise LabelParseError(f"non-numeric token in {line.strip()!r}") from exc


def read_yolo_obb_file(path: Path) -> Tuple[List[Tuple[int, List[float]]], List[str]]:
    """Read a whole YOLO-OBB ``.txt``.

    Returns ``(objects, warnings)``. Blank lines and ``#`` comments are skipped;
    malformed lines are skipped with a warning string so one bad line does not
    lose a split.
    """
    objs, warnings = [], []
    if not path.exists():
        return objs, warnings
    for lineno, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            objs.append(parse_yolo_obb_line(line))
        except LabelParseError as exc:
            warnings.append(f"{path.name}:{lineno}: {exc}")
    return objs, warnings


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def polygon_to_hbb(
    coords: Sequence[float], width: int, height: int, clip: bool = True
) -> Tuple[List[float], List[float]]:
    """Normalised 4-point polygon -> ``(bbox_xywh_px, polygon_px_flat)``.

    The bbox is the axis-aligned minimum enclosing rectangle. Both are
    optionally clipped to the image so annotations that spill a pixel past the
    border don't produce negative coordinates.
    """
    xs = [coords[i] * width for i in (0, 2, 4, 6)]
    ys = [coords[i] * height for i in (1, 3, 5, 7)]
    if clip:
        xs = [min(max(x, 0.0), float(width)) for x in xs]
        ys = [min(max(y, 0.0), float(height)) for y in ys]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    bbox = [x0, y0, x1 - x0, y1 - y0]
    poly = [v for pair in zip(xs, ys) for v in pair]
    return bbox, poly


def shoelace_area(poly_flat: Sequence[float]) -> float:
    """Area of a flat ``[x1,y1,...,xn,yn]`` polygon; winding-agnostic."""
    pts = list(zip(poly_flat[0::2], poly_flat[1::2]))
    s = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def read_image_size(path: Path) -> Tuple[int, int]:
    """``(width, height)`` from the header only -- PIL is lazy, no decode."""
    try:
        from PIL import Image  # noqa: WPS433
    except ImportError:  # pragma: no cover
        import cv2  # type: ignore
        im = cv2.imread(str(path))
        if im is None:
            raise OSError(f"could not read image: {path}")
        return im.shape[1], im.shape[0]
    with Image.open(path) as im:
        return im.size


def find_image_for_stem(
    images_dir: Path, stem: str, extensions: Sequence[str] = DEFAULT_IMAGE_EXTENSIONS
) -> Optional[Path]:
    for ext in extensions:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    lowered = {e.lower() for e in extensions}
    for p in images_dir.glob(f"{stem}.*"):
        if p.suffix.lower() in lowered:
            return p
    return None


def load_class_names(data_yaml: Path) -> List[str]:
    """``names`` from a data.yaml -- list form or ``{id: name}`` dict form."""
    import yaml  # noqa: WPS433

    with Path(data_yaml).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    names = data.get("names")
    if names is None:
        raise KeyError(f"no 'names' key in {data_yaml}")
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda k: int(k))]
    if isinstance(names, (list, tuple)):
        return [str(n) for n in names]
    raise TypeError(f"unsupported 'names' type in {data_yaml}: {type(names)!r}")


def tile_provenance(stem: str) -> Optional[Dict[str, object]]:
    """Decode ``<board>__<x0>_<y0>_<w>x<h>`` if the stem follows the tiler's pattern."""
    m = TILE_STEM_RE.match(stem)
    if not m:
        return None
    return {
        "source_board": m["board"],
        "tile_x0": int(m["x0"]),
        "tile_y0": int(m["y0"]),
        "tile_w": int(m["w"]),
        "tile_h": int(m["h"]),
    }


# ---------------------------------------------------------------------------
# Layout + configuration
# ---------------------------------------------------------------------------

@dataclass
class DatasetLayout:
    """Where the source dataset lives and what its folders are called.

    Every name can be overridden globally or per split. A per-split override
    may be a plain folder name (relative to that split's directory) or an
    absolute path.

    Examples
    --------
    ::

        DatasetLayout(root="/kaggle/working/PCB-Defect-tiles")
        DatasetLayout(root=..., splits=("train", "test"))
        DatasetLayout(root=..., split_dirs={"valid": "validation"})
        DatasetLayout(root=..., labels_dirs={"train": "/elsewhere/train_labels"})
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
        p = Path(name)
        return p if p.is_absolute() else base / p

    def split_dir(self, split: str) -> Path:
        return self._resolve(self.root, self.split_dirs.get(split, split))

    def images_dir(self, split: str) -> Path:
        return self._resolve(self.split_dir(split), self.images_dirs.get(split, self.images_dirname))

    def labels_dir(self, split: str) -> Path:
        return self._resolve(self.split_dir(split), self.labels_dirs.get(split, self.labels_dirname))

    def available_splits(self) -> List[str]:
        """Splits whose images directory exists -- image-driven, like the tiler."""
        return [s for s in self.splits if self.images_dir(s).is_dir()]

    def describe(self) -> Dict[str, Dict[str, str]]:
        return {
            s: {
                "split_dir": str(self.split_dir(s)),
                "images_dir": str(self.images_dir(s)),
                "labels_dir": str(self.labels_dir(s)),
                "exists": str(self.images_dir(s).is_dir()),
            }
            for s in self.splits
        }


@dataclass
class ConvertConfig:
    """Everything about what gets written and where."""

    # ---- where -------------------------------------------------------------
    output_root: Optional[Path] = None
    """``None`` writes each split's JSON at ``<root>/<split>/<json_name>``,
    beside that split's ``images/`` and ``labels/``. Give a path to write
    ``<output_root>/<split>/<json_name>`` instead (folders are created)."""

    json_name: str = "_annotations.coco.json"
    """Filename of the per-split JSON. Roboflow's convention by default."""

    file_name_relative_to: str = "split"
    """How ``images[].file_name`` is written:
    ``"split"``  -> ``images/<name>.jpg``   (relative to the split dir; pair with
                    MMDetection ``data_prefix=dict(img='<split>/')``)
    ``"images"`` -> ``<name>.jpg``          (relative to the images dir; pair with
                    ``data_prefix=dict(img='<split>/images/')``)
    ``"root"``   -> ``<split>/images/<name>.jpg``
    ``"absolute"`` -> full path.
    Roboflow exports use ``"images"`` semantics with images beside the JSON;
    for a tiler layout ``"images"`` plus ``data_prefix='<split>/images/'`` is
    the least surprising."""

    # ---- classes -----------------------------------------------------------
    class_names: Optional[Sequence[str]] = None
    """Explicit class list. If ``None``, read from ``data_yaml``."""

    data_yaml: Optional[Path] = None
    """Defaults to ``<root>/data.yaml``."""

    category_id_start: int = 1
    """First category id. COCO convention is 1 (0 often means background)."""

    supercategory: str = "none"
    """Value of ``categories[].supercategory`` for every class."""

    roboflow_supercategory: Optional[str] = None
    """If set, prepend a dummy category with this name at id 0 and start real
    classes at id 1 -- byte-compatible with Roboflow's COCO export shape.
    Overrides ``category_id_start``."""

    # ---- geometry ----------------------------------------------------------
    clip_to_image: bool = True
    """Clamp polygon vertices into the image before computing the box."""

    min_box_size: float = 1.0
    """Drop boxes whose width or height (px) is below this after clipping."""

    min_box_area: float = 0.0
    """Drop boxes whose area (px²) is below this. 0 disables."""

    include_segmentation: bool = True
    """Write the 4-point polygon into ``segmentation`` so the JSON is lossless.
    Detection-only loaders ignore it; set False for the smallest file."""

    area_from: str = "bbox"
    """``"bbox"`` -> area = w*h of the horizontal box (COCO detection norm).
    ``"polygon"`` -> shoelace area of the rotated polygon (tighter; matches
    what a segmentation evaluator would compute)."""

    coord_decimals: int = 2

    # ---- images ------------------------------------------------------------
    image_extensions: Sequence[str] = DEFAULT_IMAGE_EXTENSIONS

    include_empty_images: bool = True
    """Images with no label file still get an ``images[]`` entry (COCO norm).
    MMDetection then decides via ``filter_cfg``. Set False to omit them."""

    include_provenance: bool = True
    """If the stem matches the tiler's ``<board>__x_y_wxh`` pattern, add
    ``source_board``, ``tile_x0``, ``tile_y0``, ``tile_w``, ``tile_h`` to the
    image entry. Extra keys are ignored by pycocotools and make board-level
    leakage checks a one-liner."""

    # ---- bookkeeping -------------------------------------------------------
    overwrite: bool = True
    manifest_name: str = "coco_conversion_manifest.json"
    include_file_records: bool = False
    """Per-image entries in the manifest. Off by default -- tiled sets have
    thousands of images and the JSON itself already lists them."""

    dry_run: bool = False

    def __post_init__(self) -> None:
        if self.output_root is not None:
            self.output_root = Path(self.output_root)
        if self.data_yaml is not None:
            self.data_yaml = Path(self.data_yaml)
        if self.file_name_relative_to not in ("split", "images", "root", "absolute"):
            raise ValueError(f"file_name_relative_to must be split|images|root|absolute, "
                             f"got {self.file_name_relative_to!r}")
        if self.area_from not in ("bbox", "polygon"):
            raise ValueError(f"area_from must be bbox|polygon, got {self.area_from!r}")

    # ---- resolution --------------------------------------------------------
    def json_path_for(self, layout: DatasetLayout, split: str) -> Path:
        if self.output_root is None:
            return layout.split_dir(split) / self.json_name
        return self.output_root / split / self.json_name

    def manifest_dir_for(self, layout: DatasetLayout) -> Path:
        return Path(self.output_root) if self.output_root else Path(layout.root)

    def build_categories(self, names: Sequence[str]) -> Tuple[List[Dict], int]:
        """``(categories[], id_offset)`` where ``category_id = class_id + id_offset``."""
        if self.roboflow_supercategory is not None:
            cats = [dict(id=0, name=self.roboflow_supercategory, supercategory="none")]
            cats += [dict(id=i + 1, name=n, supercategory=self.roboflow_supercategory)
                     for i, n in enumerate(names)]
            return cats, 1
        start = self.category_id_start
        cats = [dict(id=i + start, name=n, supercategory=self.supercategory)
                for i, n in enumerate(names)]
        return cats, start


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class ImageRecord:
    stem: str
    file_name: str
    width: int
    height: int
    n_objects: int = 0
    n_dropped: int = 0
    warnings: List[str] = field(default_factory=list)


@dataclass
class SplitResult:
    split: str
    images_dir: str
    labels_dir: str
    json_path: str
    images: List[ImageRecord] = field(default_factory=list)
    class_counts: Dict[str, int] = field(default_factory=dict)
    n_skipped_images: int = 0

    @property
    def n_images(self) -> int:
        return len(self.images)

    @property
    def n_images_with_objects(self) -> int:
        return sum(1 for i in self.images if i.n_objects)

    @property
    def n_annotations(self) -> int:
        return sum(i.n_objects for i in self.images)

    @property
    def n_dropped(self) -> int:
        return sum(i.n_dropped for i in self.images)

    @property
    def warnings(self) -> List[str]:
        return [w for i in self.images for w in i.warnings]

    def summary(self, include_files: bool = False) -> Dict:
        out = {
            "split": self.split,
            "images_dir": self.images_dir,
            "labels_dir": self.labels_dir,
            "json_path": self.json_path,
            "n_images": self.n_images,
            "n_images_with_objects": self.n_images_with_objects,
            "n_background_images": self.n_images - self.n_images_with_objects,
            "n_images_skipped": self.n_skipped_images,
            "n_annotations": self.n_annotations,
            "n_dropped_boxes": self.n_dropped,
            "class_counts": dict(sorted(self.class_counts.items())),
            "n_warnings": len(self.warnings),
            "warnings": self.warnings[:50],
        }
        if include_files:
            out["images"] = [vars(i) for i in self.images]
        return out


# ---------------------------------------------------------------------------
# Conversion core
# ---------------------------------------------------------------------------

def _file_name(cfg: ConvertConfig, layout: DatasetLayout, split: str, image_path: Path) -> str:
    mode = cfg.file_name_relative_to
    if mode == "absolute":
        return str(image_path.resolve())
    if mode == "images":
        return image_path.name
    if mode == "split":
        return str(image_path.relative_to(layout.split_dir(split))).replace("\\", "/")
    return str(image_path.relative_to(layout.root)).replace("\\", "/")  # root


def convert_split(
    layout: DatasetLayout,
    split: str,
    class_names: Sequence[str],
    cfg: ConvertConfig,
) -> SplitResult:
    """Build and write one split's COCO JSON."""
    images_dir = layout.images_dir(split)
    labels_dir = layout.labels_dir(split)
    json_path = cfg.json_path_for(layout, split)

    if not images_dir.is_dir():
        raise FileNotFoundError(f"images directory not found for split '{split}': {images_dir}")
    if not labels_dir.is_dir():
        logger.warning("labels directory not found for split '%s': %s -- every image "
                       "will be written as background", split, labels_dir)

    result = SplitResult(split=split, images_dir=str(images_dir),
                         labels_dir=str(labels_dir), json_path=str(json_path))

    categories, id_offset = cfg.build_categories(class_names)
    images_out: List[Dict] = []
    anns_out: List[Dict] = []
    ann_id = 1
    rnd = cfg.coord_decimals

    lowered = {e.lower() for e in cfg.image_extensions}
    image_files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in lowered)
    logger.info("[%s] %d images -> %s", split, len(image_files), json_path)

    for img_id, image_path in enumerate(image_files, start=1):
        try:
            width, height = read_image_size(image_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s: could not read size (%s); skipped", image_path.name, exc)
            result.n_skipped_images += 1
            continue

        rec = ImageRecord(stem=image_path.stem,
                          file_name=_file_name(cfg, layout, split, image_path),
                          width=width, height=height)

        objs, warns = read_yolo_obb_file(labels_dir / f"{image_path.stem}.txt")
        rec.warnings.extend(warns)

        if not objs and not cfg.include_empty_images:
            result.n_skipped_images += 1
            continue

        entry: Dict = dict(id=img_id, file_name=rec.file_name, width=width, height=height)
        if cfg.include_provenance:
            prov = tile_provenance(image_path.stem)
            if prov:
                entry.update(prov)
        images_out.append(entry)

        for class_id, coords in objs:
            bbox, poly = polygon_to_hbb(coords, width, height, clip=cfg.clip_to_image)
            _, _, bw, bh = bbox
            if bw < cfg.min_box_size or bh < cfg.min_box_size:
                rec.n_dropped += 1
                continue
            area = shoelace_area(poly) if cfg.area_from == "polygon" else bw * bh
            if cfg.min_box_area > 0 and area < cfg.min_box_area:
                rec.n_dropped += 1
                continue
            if not (0 <= class_id < len(class_names)):
                rec.warnings.append(f"{image_path.stem}: class id {class_id} out of range; dropped")
                rec.n_dropped += 1
                continue

            ann: Dict = dict(
                id=ann_id,
                image_id=img_id,
                category_id=class_id + id_offset,
                bbox=[round(v, rnd) for v in bbox],
                area=round(area, rnd),
                iscrowd=0,
            )
            ann["segmentation"] = [[round(v, rnd) for v in poly]] if cfg.include_segmentation else []
            anns_out.append(ann)
            ann_id += 1
            rec.n_objects += 1
            cname = class_names[class_id]
            result.class_counts[cname] = result.class_counts.get(cname, 0) + 1

        result.images.append(rec)

    coco = dict(
        info=dict(
            description=f"{layout.root.name} / {split} -- converted from YOLO-OBB",
            version=__version__,
            date_created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            source_format="yolo-obb",
            bbox_mode="axis-aligned minimum enclosing rectangle of the oriented polygon",
        ),
        licenses=[],
        images=images_out,
        annotations=anns_out,
        categories=categories,
    )

    if cfg.dry_run:
        return result
    if json_path.exists() and not cfg.overwrite:
        logger.warning("%s exists and overwrite=False; not written", json_path)
        return result
    try:
        json_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(
            f"cannot create {json_path.parent} ({exc}). The root is not writable -- "
            f"copy the dataset under /kaggle/working or set ConvertConfig(output_root=...)."
        ) from exc
    json_path.write_text(json.dumps(coco), encoding="utf-8")

    logger.info("[%s] %d images (%d background), %d annotations, %d dropped -> %s",
                split, result.n_images, result.n_images - result.n_images_with_objects,
                result.n_annotations, result.n_dropped, json_path)
    return result


def convert_dataset(
    layout: DatasetLayout,
    cfg: Optional[ConvertConfig] = None,
    splits: Optional[Sequence[str]] = None,
) -> Dict:
    """Convert every available split and write a manifest. Returns the manifest."""
    cfg = cfg or ConvertConfig()

    if cfg.class_names is not None:
        class_names = list(cfg.class_names)
        names_source = "explicit"
    else:
        data_yaml = cfg.data_yaml or (layout.root / "data.yaml")
        class_names = load_class_names(Path(data_yaml))
        names_source = str(data_yaml)
    logger.info("class names (%s): %s", names_source, class_names)

    target = list(splits) if splits is not None else layout.available_splits()
    for s in (splits or layout.splits):
        if not layout.images_dir(s).is_dir():
            logger.warning("skipping split '%s': %s does not exist", s, layout.images_dir(s))
    if not target:
        raise FileNotFoundError(f"no usable splits under {layout.root}")

    results = [convert_split(layout, s, class_names, cfg) for s in target]

    totals_cc: Dict[str, int] = {}
    for r in results:
        for k, v in r.class_counts.items():
            totals_cc[k] = totals_cc.get(k, 0) + v

    categories, id_offset = cfg.build_categories(class_names)
    manifest = {
        "converter": {"module": "obb_to_coco", "version": __version__,
                      "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        "source_format": "yolo-obb (normalised 4-point polygons)",
        "target_format": "coco detection json (hbb = min enclosing rect; polygon in segmentation)",
        "layout": {"root": str(layout.root), "splits": list(layout.splits),
                   "images_dirname": layout.images_dirname, "labels_dirname": layout.labels_dirname,
                   "resolved": layout.describe()},
        "config": {
            "output_root": str(cfg.output_root) if cfg.output_root else None,
            "output_mode": "beside-images" if cfg.output_root is None else "separate-root",
            "json_name": cfg.json_name,
            "file_name_relative_to": cfg.file_name_relative_to,
            "category_id_start": id_offset,
            "roboflow_supercategory": cfg.roboflow_supercategory,
            "clip_to_image": cfg.clip_to_image,
            "min_box_size": cfg.min_box_size,
            "min_box_area": cfg.min_box_area,
            "include_segmentation": cfg.include_segmentation,
            "area_from": cfg.area_from,
            "include_empty_images": cfg.include_empty_images,
            "include_provenance": cfg.include_provenance,
            "dry_run": cfg.dry_run,
        },
        "class_names": class_names,
        "categories": categories,
        "totals": {
            "n_splits": len(results),
            "n_images": sum(r.n_images for r in results),
            "n_background_images": sum(r.n_images - r.n_images_with_objects for r in results),
            "n_annotations": sum(r.n_annotations for r in results),
            "n_dropped_boxes": sum(r.n_dropped for r in results),
            "n_warnings": sum(len(r.warnings) for r in results),
            "class_counts": dict(sorted(totals_cc.items())),
        },
        "splits": {r.split: r.summary(include_files=cfg.include_file_records) for r in results},
    }

    if not cfg.dry_run:
        mdir = cfg.manifest_dir_for(layout)
        mdir.mkdir(parents=True, exist_ok=True)
        mpath = mdir / cfg.manifest_name
        mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest["manifest_path"] = str(mpath)
        logger.info("manifest written to %s", mpath)
    return manifest


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_conversion(
    layout: DatasetLayout,
    cfg: ConvertConfig,
    split: str,
    tolerance_px: float = 0.5,
) -> Dict:
    """Independently re-check the written JSON against the source labels.

    Loads through pycocotools (so a structurally invalid file fails here, not
    inside a training loop), then for every image recomputes the boxes from the
    source ``.txt`` and compares. Also checks every bbox lies inside its image,
    every category id resolves, and every source label line is accounted for
    as either written or deliberately dropped.
    """
    from pycocotools.coco import COCO  # noqa: WPS433

    json_path = cfg.json_path_for(layout, split)
    labels_dir = layout.labels_dir(split)
    issues: List[str] = []

    if not json_path.exists():
        return {"split": split, "json_path": str(json_path), "ok": False,
                "issues": ["json not found"]}

    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(json_path))

    cat_ids = set(coco.getCatIds())
    names = cfg.class_names or load_class_names(cfg.data_yaml or layout.root / "data.yaml")
    _, id_offset = cfg.build_categories(names)

    max_dev = 0.0
    n_checked = 0
    n_bg = 0
    stems_seen = set()

    for img in coco.loadImgs(coco.getImgIds()):
        stem = Path(img["file_name"]).stem
        stems_seen.add(stem)
        W, H = img["width"], img["height"]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img["id"]))
        if not anns:
            n_bg += 1

        for a in anns:
            x, y, w, h = a["bbox"]
            if x < -tolerance_px or y < -tolerance_px or x + w > W + tolerance_px or y + h > H + tolerance_px:
                issues.append(f"{stem}: bbox {a['bbox']} outside {W}x{H}")
            if a["category_id"] not in cat_ids:
                issues.append(f"{stem}: category_id {a['category_id']} not in categories")
            if w <= 0 or h <= 0:
                issues.append(f"{stem}: degenerate bbox {a['bbox']}")

        # recompute from source and compare
        src, _ = read_yolo_obb_file(labels_dir / f"{stem}.txt")
        expected = []
        for cid, coords in src:
            bbox, _ = polygon_to_hbb(coords, W, H, clip=cfg.clip_to_image)
            if bbox[2] >= cfg.min_box_size and bbox[3] >= cfg.min_box_size:
                expected.append((cid + id_offset, bbox))
        if len(expected) != len(anns):
            issues.append(f"{stem}: {len(expected)} expected boxes vs {len(anns)} written")
            continue
        for (ecid, ebox), a in zip(sorted(expected, key=lambda e: (e[0], e[1][0], e[1][1])),
                                   sorted(anns, key=lambda a: (a["category_id"], a["bbox"][0], a["bbox"][1]))):
            if ecid != a["category_id"]:
                issues.append(f"{stem}: category mismatch {ecid} vs {a['category_id']}")
            dev = max(abs(e - g) for e, g in zip(ebox, a["bbox"]))
            max_dev = max(max_dev, dev)
        n_checked += 1

    # every label file should correspond to an image entry
    for lp in labels_dir.glob("*.txt") if labels_dir.is_dir() else []:
        if lp.stem not in stems_seen:
            issues.append(f"{lp.stem}: label file has no image entry in json")

    return {
        "split": split,
        "json_path": str(json_path),
        "images": len(coco.getImgIds()),
        "background_images": n_bg,
        "annotations": len(coco.getAnnIds()),
        "categories": [c["name"] for c in coco.loadCats(coco.getCatIds())],
        "images_checked": n_checked,
        "max_bbox_deviation_px": round(max_dev, 4),
        "ok": not issues and max_dev <= tolerance_px,
        "issues": issues[:20],
        "n_issues": len(issues),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_kv(pairs: Optional[Sequence[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in pairs or []:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"expected key=value, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert YOLO-OBB labels to per-split COCO JSON (horizontal boxes).",
        epilog=("By default each split's JSON is written to <root>/<split>/_annotations.coco.json, "
                "beside that split's images/ and labels/. Pass --output to write elsewhere."),
    )
    p.add_argument("--root", required=True)
    p.add_argument("--output", default=None, help="write JSONs under <output>/<split>/ instead")
    p.add_argument("--splits", nargs="*", default=["train", "valid", "test"])
    p.add_argument("--images-dirname", default="images")
    p.add_argument("--labels-dirname", default="labels")
    p.add_argument("--split-dirs", nargs="*", default=None)
    p.add_argument("--images-dirs", nargs="*", default=None)
    p.add_argument("--labels-dirs", nargs="*", default=None)
    p.add_argument("--json-name", default="_annotations.coco.json")
    p.add_argument("--file-name-relative-to", default="split",
                   choices=["split", "images", "root", "absolute"])
    p.add_argument("--data-yaml", default=None)
    p.add_argument("--class-names", nargs="*", default=None)
    p.add_argument("--category-id-start", type=int, default=1)
    p.add_argument("--supercategory", default="none")
    p.add_argument("--roboflow-supercategory", default=None,
                   help="prepend a dummy id-0 category with this name (Roboflow shape)")
    p.add_argument("--no-clip", action="store_true")
    p.add_argument("--min-box-size", type=float, default=1.0)
    p.add_argument("--min-box-area", type=float, default=0.0)
    p.add_argument("--no-segmentation", action="store_true")
    p.add_argument("--area-from", default="bbox", choices=["bbox", "polygon"])
    p.add_argument("--decimals", type=int, default=2)
    p.add_argument("--drop-empty-images", action="store_true")
    p.add_argument("--no-provenance", action="store_true")
    p.add_argument("--no-overwrite", action="store_true")
    p.add_argument("--file-records", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if a.quiet else logging.INFO,
                        format="%(levelname)s %(message)s")

    layout = DatasetLayout(
        root=Path(a.root), splits=a.splits,
        images_dirname=a.images_dirname, labels_dirname=a.labels_dirname,
        split_dirs=_parse_kv(a.split_dirs), images_dirs=_parse_kv(a.images_dirs),
        labels_dirs=_parse_kv(a.labels_dirs),
    )
    cfg = ConvertConfig(
        output_root=Path(a.output) if a.output else None,
        json_name=a.json_name,
        file_name_relative_to=a.file_name_relative_to,
        class_names=a.class_names,
        data_yaml=Path(a.data_yaml) if a.data_yaml else None,
        category_id_start=a.category_id_start,
        supercategory=a.supercategory,
        roboflow_supercategory=a.roboflow_supercategory,
        clip_to_image=not a.no_clip,
        min_box_size=a.min_box_size,
        min_box_area=a.min_box_area,
        include_segmentation=not a.no_segmentation,
        area_from=a.area_from,
        coord_decimals=a.decimals,
        include_empty_images=not a.drop_empty_images,
        include_provenance=not a.no_provenance,
        overwrite=not a.no_overwrite,
        include_file_records=a.file_records,
        dry_run=a.dry_run,
    )

    m = convert_dataset(layout, cfg)
    t = m["totals"]
    print(f"{t['n_images']} images ({t['n_background_images']} background), "
          f"{t['n_annotations']} annotations, {t['n_dropped_boxes']} dropped, "
          f"{t['n_warnings']} warnings")
    for s, summ in m["splits"].items():
        print(f"  {s} -> {summ['json_path']}  ({summ['n_images']} images, {summ['n_annotations']} boxes)")
    print("class counts:", t["class_counts"])

    if a.verify and not a.dry_run:
        for s in m["splits"]:
            r = verify_conversion(layout, cfg, s)
            print(f"[verify:{s}] {r.get('images')} images, {r.get('annotations')} anns, "
                  f"max dev {r.get('max_bbox_deviation_px')} px, ok={r['ok']}, issues={r.get('n_issues', 0)}")
            for i in r["issues"][:10]:
                print("   ", i)
    return 0


if __name__ == "__main__":
    sys.exit(main())
