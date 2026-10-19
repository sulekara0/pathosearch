"""Held-out corpus selection and leakage checks for the golden set.

The corpus is whatever a *selection rule* returns; no corpus size is known here. A new
golden version uses a new rule (or more files on disk) and the same code.

Leakage checks (all pure functions over names / vectors, tested with fakes):
    (a) :func:`overlap_with_main_ids`  golden images present in the main corpus ``ids``
    (b) :func:`train_row_violations`   golden images with a ``train`` row in the CSV
    (c) :func:`near_duplicates`        golden vs main-corpus near copies by cosine > tau
    (d) :func:`duplicate_names`, :func:`within_near_duplicates`  repeats inside the golden set
    (e) :func:`assert_disjoint`        reusable guard, e.g. for the R4 training manifest

``NEAR_DUP_TAU`` is fixed before looking at any data (R2 plan, A8); it is applied to CLIP
and QuiltNet embeddings separately.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pathosearch.data.prepare_quilt import parse_pathology
from pathosearch.eval.corpus_analysis import CsvScan, split_class
from pathosearch.eval.golden import CorpusEntry

NEAR_DUP_TAU = 0.95
_MATMUL_CHUNK = 256
_TRAIN_CLASSES = {"train_only", "train_and_val"}

# rule(split_class, in_main_ids) -> keep? Applied to images present on disk.
Rule = Callable[[str, bool], bool]
RULES: dict[str, Rule] = {
    # v0: on disk, CSV split class is val only, not part of the main corpus
    "val_only_on_disk": lambda cls, in_main: cls == "val_only" and not in_main,
}


class LeakageError(ValueError):
    """Golden images overlap with data they must be disjoint from."""


def select_heldout(
    scan: CsvScan, disk: Iterable[str], ids: Iterable[str], rule: str
) -> list[CorpusEntry]:
    """Corpus entries chosen by ``rule`` among on-disk images, sorted by image name.

    ``scan`` must come from :func:`scan_csv` called with the on-disk names, so that
    ``scan.disk_rows`` holds their rows. The category is the first ``pathology`` of the
    first CSV row of an image (same convention as the main corpus).
    """
    try:
        keep = RULES[rule]
    except KeyError:
        raise ValueError(f"unknown rule {rule!r}; known: {sorted(RULES)}") from None
    main_ids = set(ids)
    chosen = sorted(
        n
        for n in set(disk)
        if n in scan.image_mask and keep(split_class(scan.image_mask[n]), n in main_ids)
    )
    if not chosen:
        return []
    chosen_set = set(chosen)
    rows = scan.disk_rows
    labels: dict[str, str] = {}
    for name, raw in zip(rows["image_path"], rows["pathology"], strict=True):
        if name in chosen_set:
            labels.setdefault(name, parse_pathology(raw))
    missing = chosen_set - set(labels)
    if missing:
        raise ValueError(f"scan has no CSV rows for {sorted(missing)[:5]}; rescan with disk names")
    return [CorpusEntry(n, labels[n], split_class(scan.image_mask[n])) for n in chosen]


# --- (a) / (e) disjointness ----------------------------------------------------------


def overlap_with_main_ids(golden_ids: Iterable[str], main_ids: Iterable[str]) -> list[str]:
    return sorted(set(golden_ids) & set(main_ids))


def assert_disjoint(train_ids: Iterable[str], golden_ids: Iterable[str]) -> None:
    """Raise :class:`LeakageError` if any golden image is also a training image."""
    shared = overlap_with_main_ids(golden_ids, train_ids)
    if shared:
        raise LeakageError(
            f"{len(shared)} golden images are also in the training set: {shared[:5]}"
        )


# --- (b) CSV split -------------------------------------------------------------------


def train_row_violations(
    golden_ids: Iterable[str], image_mask: Mapping[str, int]
) -> dict[str, list[str]]:
    """Golden images with a ``train`` row in the CSV, and images the CSV does not know.

    Images missing from the CSV cannot be shown to be train-free, so they are reported too.
    """
    has_train: list[str] = []
    not_in_csv: list[str] = []
    for name in sorted(set(golden_ids)):
        if name not in image_mask:
            not_in_csv.append(name)
        elif split_class(image_mask[name]) in _TRAIN_CLASSES:
            has_train.append(name)
    return {"has_train_row": has_train, "not_in_csv": not_in_csv}


# --- (c) / (d) near copies -----------------------------------------------------------


@dataclass(frozen=True)
class NearDuplicate:
    image: str
    max_cosine: float
    nearest: str


def _unit(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError("embeddings must be a 2-D array")
    norm = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.maximum(norm, 1e-12)


def near_duplicates(
    golden_emb: np.ndarray,
    golden_ids: Sequence[str],
    main_emb: np.ndarray,
    main_ids: Sequence[str],
    tau: float = NEAR_DUP_TAU,
) -> list[NearDuplicate]:
    """Golden images whose max cosine to any main-corpus image exceeds ``tau``."""
    g, m = _unit(golden_emb), _unit(main_emb)
    if len(g) != len(golden_ids) or len(m) != len(main_ids):
        raise ValueError("embeddings and ids differ in length")
    if g.shape[1] != m.shape[1]:
        raise ValueError("golden and main embeddings differ in dimension")
    found: list[NearDuplicate] = []
    for start in range(0, len(g), _MATMUL_CHUNK):  # bounded memory for 100k+ main images
        sims = g[start : start + _MATMUL_CHUNK] @ m.T
        best = sims.argmax(axis=1)
        for row, j in enumerate(best):
            score = float(sims[row, j])
            if score > tau:
                found.append(NearDuplicate(golden_ids[start + row], score, main_ids[j]))
    return sorted(found, key=lambda d: (-d.max_cosine, d.image))


def duplicate_names(names: Iterable[str]) -> list[str]:
    return sorted(n for n, c in Counter(names).items() if c > 1)


def within_near_duplicates(
    emb: np.ndarray, ids: Sequence[str], tau: float = NEAR_DUP_TAU
) -> list[tuple[str, str, float]]:
    """Pairs of golden images (a < b) with cosine > ``tau``."""
    u = _unit(emb)
    if len(u) != len(ids):
        raise ValueError("embeddings and ids differ in length")
    pairs: list[tuple[str, str, float]] = []
    for start in range(0, len(u), _MATMUL_CHUNK):
        sims = u[start : start + _MATMUL_CHUNK] @ u.T
        for row in range(sims.shape[0]):
            i = start + row
            for j in np.nonzero(sims[row] > tau)[0]:
                if j > i:
                    a, b = sorted((ids[i], ids[int(j)]))
                    pairs.append((a, b, float(sims[row, j])))
    return sorted(pairs)


# --- B1 / B3 helpers: manifest metadata, comparison, aggregate report ------------------


def manifest_meta(
    rule: str,
    entries: Sequence[CorpusEntry],
    source_files: Mapping[str, Path],
    disk_images: int,
) -> dict[str, Any]:
    """Provenance record written next to the manifest (rule, source size/mtime, counts)."""
    sources: dict[str, Any] = {}
    for label, path in source_files.items():
        st = path.stat()
        sources[label] = {
            "path": path.as_posix(),
            "size_bytes": st.st_size,
            "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(
                timespec="seconds"
            ),
        }
    return {
        "rule": rule,
        "images": len(entries),
        "disk_images_scanned": disk_images,
        "per_category": dict(sorted(Counter(e.pathology for e in entries).items())),
        "split_classes": dict(sorted(Counter(e.split_class for e in entries).items())),
        "sources": sources,
    }


def compare_with_analysis(entries: Sequence[CorpusEntry], analysis: Mapping[str, Any]) -> list[str]:
    """Differences between a manifest and ``reports/corpus_analysis.json`` (STATUS §7.8.1).

    Empty list = the counts match exactly. Only meaningful for the ``val_only_on_disk`` rule.
    """
    section = analysis["sec_7_8_val_only_on_disk"]
    expected = {
        cat: v["images"] for cat, v in section["all_val_only_images"]["per_category"].items()
    }
    got = Counter(e.pathology for e in entries)
    problems: list[str] = []
    if len(entries) != section["images"]:
        problems.append(f"images: manifest {len(entries)} vs analysis {section['images']}")
    for cat in sorted(set(expected) | set(got)):
        if expected.get(cat, 0) != got.get(cat, 0):
            problems.append(f"{cat}: manifest {got.get(cat, 0)} vs analysis {expected.get(cat, 0)}")
    return problems


Embeddings = Mapping[str, tuple[np.ndarray, Sequence[str]]]  # encoder -> (matrix, ids)


def build_leakage_report(
    golden_ids: Sequence[str],
    main_ids: Sequence[str],
    image_mask: Mapping[str, int],
    golden_emb: Embeddings,
    main_emb: Embeddings,
    tau: float = NEAR_DUP_TAU,
) -> dict[str, Any]:
    """Run checks (a)-(d) and summarise; ``passed`` is True only if every list is empty."""
    if set(golden_emb) != set(main_emb):
        raise ValueError("golden and main embeddings need the same encoders")
    for enc, (emb, ids) in golden_emb.items():
        if set(ids) != set(golden_ids) or len(ids) != len(set(ids)):
            raise ValueError(f"{enc}: golden embedding ids differ from the manifest images")
        if len(emb) != len(ids):
            raise ValueError(f"{enc}: golden embeddings and ids differ in length")

    near: dict[str, list[dict[str, Any]]] = {}
    within: dict[str, list[list[Any]]] = {}
    for enc in sorted(golden_emb):
        g_emb, g_ids = golden_emb[enc]
        m_emb, m_ids = main_emb[enc]
        near[enc] = [
            {"image": d.image, "max_cosine": round(d.max_cosine, 4), "nearest": d.nearest}
            for d in near_duplicates(g_emb, g_ids, m_emb, m_ids, tau)
        ]
        within[enc] = [[a, b, round(c, 4)] for a, b, c in within_near_duplicates(g_emb, g_ids, tau)]

    train = train_row_violations(golden_ids, image_mask)
    report: dict[str, Any] = {
        "tau": tau,
        "images": len(set(golden_ids)),
        "a_overlap_with_main_ids": overlap_with_main_ids(golden_ids, main_ids),
        "b_has_train_row": train["has_train_row"],
        "b_not_in_csv": train["not_in_csv"],
        "c_near_duplicates_of_main_corpus": near,
        "d_duplicate_names": duplicate_names(golden_ids),
        "d_within_golden_near_duplicates": within,
    }
    counts = {
        key: sum(len(x) for x in val.values()) if isinstance(val, dict) else len(val)
        for key, val in report.items()
        if key[:2] in {"a_", "b_", "c_", "d_"}
    }
    report["violation_counts"] = counts
    report["passed"] = not any(counts.values())
    return report
