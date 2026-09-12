# pcbkit
# pcbkit / going_modular

Dataset preparation for rotated-object detection on PCB defect boards.

Two modules, meant to be run in order:

| Module | Input | Output |
| --- | --- | --- |
| `tile_obb_dataset.py` | full-size boards + YOLO-OBB labels | overlapping tiles + YOLO-OBB labels |
| `yolo_obb_to_dota.py` | YOLO-OBB labels | DOTA annotations, written alongside |

The tiler solves a resolution problem. The converter solves a format problem.
They are independent — you can use either alone — but if you want DOTA
annotations for tiles, tile **first**. DOTA coordinates are absolute pixels
tied to one specific image, so annotations made for a full board do not
transfer to crops of that board.

---

## Install

```python
import subprocess, sys
from pathlib import Path

REPO = Path("pcbkit")
if not REPO.exists():
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/jawadrashidd/pcbkit.git", str(REPO)],
        check=True,
    )
    print("pcbkit is cloned")
else:
    print("pcbkit already present")

sys.path.insert(0, str((REPO / "going_modular").resolve()))
```

Pointing `sys.path` at `going_modular/` lets you `from tile_obb_dataset import ...`.
Point it at the repo root instead if you prefer `from going_modular.tile_obb_dataset import ...`.

Requirements: `opencv-python`, `numpy`, `shapely`, `pillow`, `pyyaml`.

---

## Quick start

```python
from tile_obb_dataset import DatasetLayout, TileConfig, tile_dataset, verify_tiling
from yolo_obb_to_dota import DatasetLayout as ConvLayout, ConvertConfig, convert_dataset

# 1. tile the boards
layout = DatasetLayout(root="/kaggle/working/PCB-Defect-6")
cfg = TileConfig(
    output_root="/kaggle/working/PCB-Defect-tiles",
    tile=1024, overlap=200, bg_keep=0.7, val_split="valid",
)
manifest = tile_dataset(layout, cfg)

for split in manifest["splits"]:
    print(verify_tiling(layout, cfg, split))

# 2. add DOTA annotations beside the tiles
conv_layout = ConvLayout(root="/kaggle/working/PCB-Defect-tiles")
conv_cfg = ConvertConfig()          # output_root=None -> in-place ann/
convert_dataset(conv_layout, conv_cfg)
```

Result:

```
PCB-Defect-tiles/
    data.yaml                    for Ultralytics
    tiling_manifest.json
    conversion_manifest.json
    train/  images/  labels/  ann/
    valid/  images/  labels/  ann/
    test/   images/  labels/  ann/
```

`labels/` is YOLO-OBB (normalised, class index). `ann/` is DOTA
(absolute pixels, class name). Train Ultralytics off `images/` + `labels/`,
MMRotate off `images/` + `ann/`.

---

# Module 1 — `tile_obb_dataset.py`

## The problem

PCB boards are around 2600x2500 px and defects are 50-150 px. Training at
`imgsz=1024` on whole boards downscales a defect to a handful of pixels.
Tiling keeps source resolution: each crop is fed to the model 1:1, so the
defect keeps its original size on screen.

Naive grid tiling breaks this. Some defects land on a cut line, leaving half
in one tile and half in the next. Labelling the half teaches the model that a
fragment is a whole object; not labelling it teaches the model that a visible
defect is background. Both corrupt training.

## The approach

Tile positions are chosen from the defects, not from a fixed spacing.

**Anchor tiles.** For each not-yet-covered object, find a window that contains
it completely and slices no other object. Candidate offsets are the centred
position plus, for every object edge, the two offsets that put a tile boundary
just outside it — precisely where a cutting window flips to a clean one. So a
few dozen candidates per axis cover the whole solution space rather than
thousands of pixel offsets. Any other objects that land whole inside the chosen
tile are marked covered too, so a cluster of five nearby defects costs one tile
rather than five.

**Grid tiles.** A regular overlapping grid supplies background and context. Any
position that would slice an object is discarded. Object-free ones are
subsampled at `bg_keep`.

**Guarantee.** Every object appears complete in at least one tile, and no tile
anywhere contains a partial object.

**Shrink fallback.** When two objects sit close enough that every full-size
window containing one must slice the other, the anchor falls back to a smaller
crop — 3/4, 1/2, 3/8, 1/4 of `tile` — until it can isolate cleanly. Those crops
are letterboxed by YOLO at load time. The run reports how many were shrunk.

## Public API

### `DatasetLayout`

Where the source lives and what the folders are called.

| Field | Default | Meaning |
| --- | --- | --- |
| `root` | required | dataset root holding the split folders |
| `splits` | `("train", "valid", "test")` | split names to look for |
| `images_dirname` | `"images"` | images folder name inside each split |
| `labels_dirname` | `"labels"` | labels folder name inside each split |
| `split_dirs` | `{}` | per-split folder overrides, e.g. `{"valid": "validation"}` |
| `images_dirs` | `{}` | per-split images overrides |
| `labels_dirs` | `{}` | per-split labels overrides |

Per-split overrides accept a plain name (resolved relative to the split
directory) or an absolute path.

| Method | Returns |
| --- | --- |
| `split_dir(split)` | resolved path to that split |
| `images_dir(split)` | resolved images path |
| `labels_dir(split)` | resolved labels path |
| `available_splits()` | splits whose images directory exists on disk |
| `describe()` | dict of resolved paths per split — print this before a long run |

Missing splits are skipped with a warning, not an error. A two-split dataset
works with the default `splits` unchanged.

### `TileConfig`

| Field | Default | Meaning |
| --- | --- | --- |
| `output_root` | `"PCB-Defect-tiles"` | where tiles go; must be writable |
| `tile` | `1024` | crop size in source pixels; train with `imgsz=tile` |
| `overlap` | `200` | overlap between **grid** tiles only |
| `bg_keep` | `0.3` | fraction of object-free grid tiles to keep |
| `shrink_steps` | `(1.0, .75, .5, .375, .25)` | crop-size fractions tried when isolation fails |
| `seed` | `0` | seeds background subsampling; makes runs reproducible |
| `jpeg_quality` | `98` | encoder quality; high, because artefacts are defect-sized |
| `write_data_yaml` | `True` | emit a `data.yaml` for the tiled set |
| `data_yaml` | `None` | source yaml; defaults to `<root>/data.yaml` |
| `val_split` | `"test"` | which split the generated yaml points `val` at |
| `manifest_name` | `"tiling_manifest.json"` | |
| `include_file_records` | `True` | store a per-board row in the manifest |
| `dry_run` | `False` | plan and count without writing anything |

Validated on construction: `overlap` must be in `[0, tile)` and `bg_keep` in
`[0, 1]`, otherwise `ValueError`.

Property `step` is `tile - overlap`, the distance between grid origins.

**On `overlap`.** It no longer guards against sliced objects — anchor placement
does that at any value, including zero. Treat it as a background
sampling-density knob. Raise it if many grid tiles are being dropped.

**On `bg_keep`.** Defect tiles are already 99%+ background pixels, so dedicated
background tiles are not what teaches "this trace is fine" — they add scenery
you would otherwise never see. Standard guidance keeps background images a
minority; roughly one background tile per two or three defect tiles is a
reasonable target. On dense boards the flag barely matters, since most grid
positions are discarded for slicing something before the dice roll happens.

### `tile_dataset(layout, cfg=None, splits=None) -> dict`

Tiles the whole dataset, writes a JSON manifest, and returns it. `splits`
restricts the run; defaults to every split whose images directory exists.

Manifest sections:

- `config` — every setting used, including `seed`, so the run is reproducible
- `layout` — resolved source paths, so you know which export produced these tiles
- `totals` — tiles, anchors, background, objects, instances, grid tiles dropped, anchors shrunk
- `splits[<name>]` — the same counts per split, plus `images`: one row per source board
- `issues` — actionable warnings, empty on a clean run

### `verify_tiling(layout, cfg, split, iou_tolerance=0.97) -> dict`

Independently re-checks both guarantees against what is actually on disk. For
every written tile it reparses the source labels, recomputes which objects the
crop window sees, asserts none is partial and that the written label count
matches, then reprojects each written box back to board coordinates and
compares it to the original polygon by IoU. Finally it checks that every source
object appears whole in at least one tile.

Returns `tiles_checked`, `objects`, `objects_covered`, `worst_box_iou`, `ok`,
and a list of `issues`. Run it once after tiling; it is cheap relative to
training.

## Internal functions

Useful if you are modifying the tiler.

| Function | Role |
| --- | --- |
| `read_yolo_obb_file(path, w, h)` | parse a label file into pixel-space shapely polygons; returns `(objects, warnings)`. Malformed lines are skipped with a warning, not raised. Self-intersecting corner orders are repaired with `buffer(0)` |
| `obb_line(class_id, poly, x0, y0, tw, th)` | format one polygon as a YOLO-OBB line normalised against the tile |
| `tile_name(stem, x0, y0, tw, th)` | build the provenance-encoding filename |
| `window(x0, y0, w, h, tile)` | clip a tile origin to the image |
| `classify(win, objs)` | `(indices fully inside, any cut?)`; returns early on the first cut |
| `axis_candidates(lo, hi, tile, length, edges)` | candidate start offsets on one axis |
| `try_size(...)` | search candidates for a clean window at one exact size |
| `place_anchor(...)` | `try_size` across the shrink steps |
| `grid_starts(length, tile, step)` | regular offsets; last one flush with the image edge |
| `plan_tiles(objs, w, h, cfg, rng, result)` | full placement for one board |
| `tile_image(...)` | tile one board and write its crops |
| `tile_split(layout, split, cfg, rng)` | tile every board in one split |
| `write_data_yaml(layout, cfg, splits)` | copy `names` from the source yaml into a tiled one |

Constants `WHOLE = 0.999` and `TOUCH = 1e-9` are the area fractions at which an
object counts as fully inside, or absent, respectively.

## Output

```
<output_root>/<split>/images/<stem>__<x0>_<y0>_<w>x<h>.jpg
<output_root>/<split>/labels/<stem>__<x0>_<y0>_<w>x<h>.txt
```

The filename encodes source board, offset, and crop size, so a tile can always
be traced back. The size suffix matters because a shrunk anchor and a full tile
can share an origin.

A label file is written **only** when the tile contains at least one object —
the Ultralytics convention for background images. So
`len(images) - len(labels)` is your background tile count.

## CLI

```
python tile_obb_dataset.py --root <dataset_root> --output <dest> \
    --tile 1024 --overlap 200 --bg-keep 0.7 --verify
```

| Flag | Default |
| --- | --- |
| `--root` | required |
| `--output` | required |
| `--splits` | `train valid test` |
| `--tile` | `1024` |
| `--overlap` | `200` |
| `--bg-keep` | `0.3` |
| `--no-shrink` | off — never shrink; drop uncoverable objects instead |
| `--seed` | `0` |
| `--jpeg-quality` | `98` |
| `--val-split` | `test` |
| `--no-data-yaml`, `--no-file-records`, `--dry-run`, `--verify`, `--quiet` | off |
| `--images-dirname`, `--labels-dirname` | `images`, `labels` |
| `--split-dirs`, `--images-dirs`, `--labels-dirs` | `key=value` pairs |

---

# Module 2 — `yolo_obb_to_dota.py`

## What it does

Converts YOLO-OBB labels to DOTA annotations, leaving the source `labels/`
untouched.

```
IN   <class_id> <x1> <y1> ... <x4> <y4>     9 tokens, normalised [0,1]
OUT  <x1> <y1> ... <x4> <y4> <name> <diff>  10 tokens, absolute pixels
```

Because YOLO-OBB is normalised and DOTA is pixels, image dimensions are
required. They are read from the image header via PIL — no full decode — so
conversion stays fast on large boards.

Class names come from `names` in `data.yaml`. Spaces are replaced with `-`,
since DOTA lines are space-delimited.

## Public API

`DatasetLayout` is the same shape as the tiler's, except `available_splits()`
keys off the **labels** directory rather than images. The converter is driven
by label files; the tiler is driven by image files. Keep the two folders in
sync or they will disagree about what exists.

### `ConvertConfig`

| Field | Default | Meaning |
| --- | --- | --- |
| `output_root` | `None` | `None` writes `<root>/<split>/ann/`, beside the split's images and labels. Set a path to redirect to `<output_root>/<split>/ann/` |
| `ann_dirname` | `"ann"` | name of the created folder; `labels/` is never touched |
| `class_names` | `None` | explicit list; otherwise read from `data_yaml` |
| `data_yaml` | `None` | defaults to `<root>/data.yaml` |
| `clip_to_image` | `True` | clamp vertices into `[0,1]` before scaling |
| `coord_decimals` | `1` | decimal places in output; `0` writes integers |
| `difficult` | `0` | value in the DOTA difficult column |
| `write_dota_header` | `False` | prepend `imagesource`/`gsd` lines |
| `keep_empty` | `True` | write an empty `.txt` for object-free images so negatives stay in the set |
| `min_area_px` | `0.0` | drop boxes smaller than this after scaling; `0` disables |
| `default_image_size` | `None` | fallback when the image is missing; `None` skips such labels rather than guessing |
| `overwrite` | `True` | |
| `manifest_name` | `"conversion_manifest.json"` | |
| `include_file_records` | `True` | per-file entry in the manifest |
| `dry_run` | `False` | |

The default in-place mode needs `root` to be writable, so it fails on
`/kaggle/input`. Copy the dataset into `/kaggle/working` first, or set
`output_root`. The module warns up front when it detects this.

| Method | Returns |
| --- | --- |
| `ann_dir_for(layout, split)` | where that split's DOTA files go |
| `manifest_dir_for(layout)` | where the manifest goes |

Both conversion and verification route through these, so they cannot disagree.

### `convert_dataset(layout, cfg=None, splits=None) -> dict`

Converts and writes a manifest. Manifest carries `config` (including
`output_mode`), resolved `layout`, `class_names`, `totals` with per-class
counts, and `splits[<name>]` with `image_size_stats` and optional per-file
records.

### `verify_conversion(layout, cfg, split, tolerance_px=1.0) -> dict`

Round-trip check. Reads each converted file back, rescales using the image
size, and compares to the source values. Returns `files_checked`,
`max_deviation_px`, `within_tolerance`, and `issues`. Deviation should land
well under 1 px; anything larger means a label was measured against the wrong
image.

Note it iterates **label files**, so background tiles with no `.txt` are not
counted. `files_checked` will be your defect-bearing tile count, not the total.

## Internal functions

| Function | Role |
| --- | --- |
| `OrientedBox` | frozen 4-point box; `to_pixels`, `clipped`, `area` (shoelace, winding-agnostic), `bounds` |
| `parse_yolo_obb_line(line)` | one line to an `OrientedBox`; raises `LabelParseError` |
| `read_yolo_obb_file(path)` | whole file; returns `(boxes, warnings)` |
| `format_dota_line(box, name, decimals)` | one pixel-space box as a DOTA line |
| `read_dota_file(path)` | parse DOTA back in, for verification |
| `sanitize_class_name(name)` | spaces to hyphens |
| `read_image_size(path)` | header-only dimensions via PIL, cv2 fallback |
| `find_image_for_stem(dir, stem, exts)` | match a label to its image, case-insensitively |
| `load_class_names(data_yaml)` | handles both list and dict `names` forms |
| `resolve_class_name(names, id)` | falls back to `class_<id>` when out of range |
| `convert_boxes_to_dota_lines(...)` | normalised boxes to text lines |
| `convert_label_file(...)` | one file |
| `convert_split(...)` | one split |

## CLI

```
python yolo_obb_to_dota.py --root /kaggle/working/PCB-Defect-tiles --verify
```

Omit `--output` for the in-place layout. Pass it only to redirect.

Other flags: `--ann-dirname`, `--decimals`, `--difficult`, `--min-area-px`,
`--no-clip`, `--drop-empty`, `--dota-header`, `--class-names`, `--data-yaml`,
`--no-file-records`, `--dry-run`, `--quiet`, plus the same layout overrides as
the tiler.

---

## Notes and gotchas

**Order matters.** Tile before converting. DOTA coordinates are absolute pixels
bound to one image; annotations for a full board are meaningless for crops of
it.

**Object larger than the tile.** Cannot be enclosed. Reported as
`n_oversize_objects` and dropped. Raise `tile`.

**Object that cannot be isolated at any shrink step.** Reported as
`n_unplaceable_objects` and dropped. Also means raise `tile`. Both counters
being zero is the signal that your tile size is comfortable.

**Repaired polygons.** A self-intersecting corner order is fixed with
`buffer(0)`, which may yield more than four vertices. The written box is then a
`minAreaRect` refit, which is a genuine shape change. `verify_tiling` surfaces
these as IoU drift below `iou_tolerance`. A handful is normal on a
Roboflow export; many means the source annotations need attention.

**Duplicate boards.** Roboflow augmentation can put the same board in the
dataset under several hashes. Check the manifest's per-board records before
trusting a validation score — the same board in train and valid is leakage.

**Inference tiling is a separate problem.** At inference you have no
annotations to place tiles around, so you are back to a blind grid. Overlap
there must exceed your largest object; shift predictions into full-image
coordinates and merge with NMS.

**Editing the repo.** Python caches imported modules. After re-cloning, restart
the kernel — `sys.path` changes will not pick up new code otherwise.
