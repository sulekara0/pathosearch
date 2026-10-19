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
