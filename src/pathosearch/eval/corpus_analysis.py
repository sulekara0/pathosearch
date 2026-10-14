"""Read-only audit of the Quilt-1M subset: CSV splits, JSONL captions, images on disk.

Reproduces the counts in ``docs/STATUS.md`` (sections 7.1-7.8) from the raw files.
Nothing is written except the JSON the caller dumps; no data file is modified and
no archive (zip) is opened. Images are matched by file name (CSV ``image_path`` is a
basename).
"""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from pathosearch.data.prepare_quilt import BALANCED_TARGETS, parse_pathology
from pathosearch.labels import normalize_pathology

CHUNK_SIZE = 200_000
MIN_PER_CATEGORY = 20
_EMPTY = {"", "[]", "nan", "none"}
_SPLIT_BITS = {"train": 1, "val": 2}
_OTHER_BIT = 4
_CLASS_NAMES = {1: "train_only", 2: "val_only", 3: "train_and_val"}


def is_missing(value: object) -> bool:
    """True for NaN, empty string and the empty-list literal ``[]``."""
    if value is None or value is pd.NA or (isinstance(value, float) and value != value):
        return True
    return str(value).strip().lower() in _EMPTY


def split_class(mask: int) -> str:
    """Name of the split-membership class for a bitmask (train=1, val=2, other=4)."""
    return _CLASS_NAMES.get(mask, "other_or_mixed")


def list_image_names(images_dir: Path) -> tuple[set[str], int]:
    """File names under ``images_dir`` (recursive) and how many names repeat."""
    names: set[str] = set()
    dupes = 0
    for _root, _dirs, files in os.walk(images_dir):
        for f in files:
            if f in names:
                dupes += 1
            names.add(f)
    return names, dupes


@dataclass
class CsvScan:
    """One pass over the lookup CSV."""

    n_rows: int = 0
    columns: list[str] = field(default_factory=list)
    split_rows: Counter[str] = field(default_factory=Counter)
    image_mask: dict[str, int] = field(default_factory=dict)
    disk_rows: pd.DataFrame = field(default_factory=pd.DataFrame)


def scan_csv(csv_path: Path, keep_names: set[str]) -> CsvScan:
    """Stream the CSV; keep full rows only for images in ``keep_names``."""
    scan = CsvScan()
    kept: list[pd.DataFrame] = []
    for chunk in pd.read_csv(csv_path, dtype=str, chunksize=CHUNK_SIZE, low_memory=False):
        if not scan.columns:
            scan.columns = [str(c) for c in chunk.columns]
        scan.n_rows += len(chunk)
        splits = chunk["split"].fillna("<nan>")
        scan.split_rows.update(splits.tolist())
        for name, sp in zip(chunk["image_path"].tolist(), splits.tolist(), strict=True):
            bit = _SPLIT_BITS.get(sp, _OTHER_BIT)
            scan.image_mask[name] = scan.image_mask.get(name, 0) | bit
        hit = chunk[chunk["image_path"].isin(keep_names)]
        if len(hit):
            kept.append(hit)
    scan.disk_rows = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()
    return scan


def _class_counts(names: Iterable[str], masks: dict[str, int]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for n in names:
        c["not_in_csv" if n not in masks else split_class(masks[n])] += 1
    return dict(sorted(c.items()))


def csv_split_section(scan: CsvScan) -> dict[str, Any]:
    """7.7.1: rows and unique images per CSV split value."""
    per_split_images: Counter[str] = Counter()
    for mask in scan.image_mask.values():
        for sp, bit in _SPLIT_BITS.items():
            if mask & bit:
                per_split_images[sp] += 1
    return {
        "rows": scan.n_rows,
        "unique_images": len(scan.image_mask),
        "rows_per_split": dict(sorted(scan.split_rows.items())),
        "images_with_at_least_one_row_in_split": dict(sorted(per_split_images.items())),
        "image_split_classes": dict(
            sorted(Counter(split_class(m) for m in scan.image_mask.values()).items())
        ),
    }


def disk_section(disk: set[str], dupes: int, ids: list[str], scan: CsvScan) -> dict[str, Any]:
    """7.7.2-7.7.4: indexed ids and disk images against the CSV split classes."""
    id_set = set(ids)
    on_disk_in_ids = disk & id_set
    on_disk_not_in_ids = disk - id_set
    return {
        "ids_total": len(ids),
        "ids_unique": len(id_set),
        "ids_split_classes": _class_counts(id_set, scan.image_mask),
        "disk_unique_names": len(disk),
        "disk_duplicate_names": dupes,
        "disk_in_ids": len(on_disk_in_ids),
        "disk_not_in_ids": len(on_disk_not_in_ids),
        "ids_missing_on_disk": len(id_set - disk),
        "disk_not_in_csv": len([n for n in disk if n not in scan.image_mask]),
        "disk_split_classes": _class_counts(disk, scan.image_mask),
        "disk_not_in_ids_split_classes": _class_counts(on_disk_not_in_ids, scan.image_mask),
    }


def jsonl_section(captions_path: Path, ids: list[str]) -> dict[str, Any]:
    """7.2-7.4, 7.7.5: JSONL rows, caption multiplicity, category distribution."""
    rows_per_image: Counter[str] = Counter()
    label_of: dict[str, str] = {}
    label_conflicts = 0
    split_values: Counter[str] = Counter()
    rows_per_label: Counter[str] = Counter()
    n_rows = 0
    with captions_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            name = Path(str(rec["path"])).name
            label = normalize_pathology(rec.get("pathology")) or "unknown"
            n_rows += 1
            rows_per_image[name] += 1
            rows_per_label[label] += 1
            split_values[str(rec.get("split"))] += 1
            if label_of.setdefault(name, label) != label:
                label_conflicts += 1
    images_per_label = Counter(label_of.values())
    n_img = len(rows_per_image)
    targets = {
        k: {
            "target_rows": t,
            "jsonl_rows": rows_per_label.get(k, 0),
            "jsonl_images": images_per_label.get(k, 0),
        }
        for k, t in BALANCED_TARGETS.items()
    }
    return {
        "rows": n_rows,
        "unique_images": n_img,
        "split_field_values": dict(sorted(split_values.items())),
        "rows_per_image_histogram": {
            str(k): v for k, v in sorted(Counter(rows_per_image.values()).items())
        },
        "label_conflicts_within_image": label_conflicts,
        "ids_not_in_jsonl": len(set(ids) - set(rows_per_image)),
        "image_label_distribution": {
            k: {"images": v, "share": round(v / n_img, 4)}
            for k, v in images_per_label.most_common()
        },
        "balanced_targets_vs_jsonl": targets,
        "balanced_targets_total": sum(BALANCED_TARGETS.values()),
    }


def val_only_section(
    scan: CsvScan, disk: set[str], ids: list[str], min_per_category: int = MIN_PER_CATEGORY
) -> dict[str, Any]:
    """7.8: images on disk whose only CSV split is ``val`` (candidate held-out set)."""
    id_set = set(ids)
    names = sorted(n for n in disk if scan.image_mask.get(n) == _SPLIT_BITS["val"])
    rows = (
        scan.disk_rows[scan.disk_rows["image_path"].isin(names)] if len(names) else scan.disk_rows
    )
    n_img = len(names)

    labels: dict[str, str] = {}
    conflicts = 0
    for name, raw in zip(rows["image_path"], rows["pathology"], strict=True):
        label = parse_pathology(raw)
        if labels.setdefault(name, label) != label:
            conflicts += 1

    eligible_flag = (pd.to_numeric(rows["not_histology"], errors="coerce") == 0) & ~rows[
        "caption"
    ].map(is_missing)
    eligible = set(rows.loc[eligible_flag, "image_path"])

    def dist(members: Iterable[str]) -> dict[str, Any]:
        members = list(members)
        c = Counter(labels[m] for m in members)
        total = len(members)
        by_cat = {
            k: {"images": v, "share": round(v / total, 4) if total else 0.0}
            for k, v in c.most_common()
        }
        usable = sorted(k for k, v in c.items() if k != "unknown" and v >= min_per_category)
        return {
            "images": total,
            "unknown_images": c.get("unknown", 0),
            "unknown_share": round(c.get("unknown", 0) / total, 4) if total else 0.0,
            "per_category": by_cat,
            f"categories_with_at_least_{min_per_category}_excluding_unknown": usable,
            "categories_below_threshold": sorted(
                k for k, v in c.items() if k != "unknown" and v < min_per_category
            ),
        }

    per_image_rows = rows.groupby("image_path").size()
    distinct_captions = rows.groupby("image_path")["caption"].nunique()
    availability = {}
    for col in scan.columns:
        if col in ("image_path", "") or col.startswith("Unnamed"):
            continue
        present = ~rows[col].map(is_missing)
        availability[col] = {
            "rows_non_missing": int(present.sum()),
            "images_with_any_non_missing": int(rows.loc[present, "image_path"].nunique()),
        }

    def value_counts(col: str) -> dict[str, int]:
        return {str(k): int(v) for k, v in rows[col].fillna("<missing>").value_counts().items()}

    return {
        "definition": "on disk AND CSV split classes == val only (no train row)",
        "min_per_category": min_per_category,
        "images": n_img,
        "csv_rows": len(rows),
        "also_in_ids": len(set(names) & id_set),
        "label_conflicts_within_image": conflicts,
        "all_val_only_images": dist(labels),
        "eligible_images_not_histology0_and_caption_present": dist(
            n for n in labels if n in eligible
        ),
        "csv_rows_per_image_histogram": {
            str(k): int(v) for k, v in sorted(Counter(per_image_rows.tolist()).items())
        },
        "distinct_captions_per_image_histogram": {
            str(k): int(v) for k, v in sorted(Counter(distinct_captions.tolist()).items())
        },
        "csv_columns": scan.columns,
        "column_availability": availability,
        "magnification_values": value_counts("magnification"),
        "subset_values": value_counts("subset"),
        "not_histology_values": value_counts("not_histology"),
        "single_wsi_values": value_counts("single_wsi"),
    }


def run_analysis(
    csv_path: Path,
    captions_path: Path,
    images_dir: Path,
    ids_path: Path,
    min_per_category: int = MIN_PER_CATEGORY,
) -> dict[str, Any]:
    """Run every section; the result is JSON-serialisable."""
    ids: list[str] = json.loads(ids_path.read_text(encoding="utf-8"))
    disk, dupes = list_image_names(images_dir)
    scan = scan_csv(csv_path, disk)
    return {
        "inputs": {
            "csv": csv_path.as_posix(),
            "captions": captions_path.as_posix(),
            "images_dir": images_dir.as_posix(),
            "ids": ids_path.as_posix(),
            "csv_columns": scan.columns,
        },
        "sec_7_7_1_csv_split": csv_split_section(scan),
        "sec_7_x_jsonl": jsonl_section(captions_path, ids),
        "sec_7_7_2_to_7_7_4_disk_and_ids": disk_section(disk, dupes, ids, scan),
        "sec_7_8_val_only_on_disk": val_only_section(scan, disk, ids, min_per_category),
    }
