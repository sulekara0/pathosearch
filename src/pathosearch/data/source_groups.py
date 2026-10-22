"""Source-group keys parsed from Quilt-1M image file names, and golden/train overlap.

Images from the same source (a YouTube video, a tweet, a PubMed figure) can look alike
without being near copies by cosine. The key groups them so that a split or an exclusion
can work on whole sources. Keys are parsed from file names only; the CSV has no source
column, so the meaning of a key (e.g. "same PubMed figure") is an assumption.

    quilt     ``<11-char video id>_image_<uuid>.jpg``  -> ``video:<id>``
    openpath  ``<tweet id>_<k>.jpg|png``               -> ``tweet:<id>``
    pubmed    ``<uuid>.jpg`` or ``<uuid>_<k>.jpg``     -> ``figure:<uuid>``
    laion / anything else                              -> no key (the image is its own group)
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from pathosearch.eval.corpus_analysis import split_class

_PATTERNS = (
    ("video", re.compile(r"^([A-Za-z0-9_-]{11})_image_")),
    ("tweet", re.compile(r"^(\d+)_\d+\.(?:jpe?g|png)$", re.IGNORECASE)),
    (
        "figure",
        re.compile(
            r"^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})(?:_\d+)?\.jpe?g$",
            re.IGNORECASE,
        ),
    ),
)
KINDS = tuple(kind for kind, _ in _PATTERNS)
_EXAMPLES = 5


def source_key(name: str) -> str | None:
    """``<kind>:<id>`` for a file name, or None when no source can be parsed."""
    for kind, pat in _PATTERNS:
        m = pat.match(name)
        if m:
            return f"{kind}:{m.group(1).lower() if kind == 'figure' else m.group(1)}"
    return None


def group_key(name: str) -> str:
    """Source key, or ``image:<name>`` so that keyless images form singleton groups."""
    return source_key(name) or f"image:{name}"


def key_kind(key: str) -> str:
    return key.partition(":")[0]


def sharing_source(candidates: Iterable[str], reference: Iterable[str]) -> list[str]:
    """Candidates (sorted) whose source key also occurs among ``reference`` images."""
    ref_keys = {k for k in map(source_key, reference) if k}
    return sorted(n for n in set(candidates) if source_key(n) in ref_keys)


def _by_kind(names: Iterable[str]) -> dict[str, Any]:
    keys = [source_key(n) for n in names]
    images = Counter(key_kind(k) if k else "none" for k in keys)
    distinct = Counter(key_kind(k) for k in set(keys) if k)
    return {
        "images": sum(images.values()),
        "images_by_kind": dict(sorted(images.items())),
        "distinct_keys_by_kind": dict(sorted(distinct.items())),
    }


def _group_sizes(names: Iterable[str]) -> dict[str, dict[str, int]]:
    sizes: dict[str, Counter[str]] = defaultdict(Counter)
    for n in names:
        k = source_key(n)
        if k:
            sizes[key_kind(k)][k] += 1
    out: dict[str, dict[str, int]] = {}
    for kind, c in sorted(sizes.items()):
        v = sorted(c.values())
        out[kind] = {"groups": len(v), "median": v[len(v) // 2], "max": v[-1]}
    return out


def csv_split_by_source(image_mask: Mapping[str, int]) -> dict[str, Any]:
    """Per kind: how many source groups are train-only, val-only or mixed in the CSV."""
    masks: dict[str, int] = defaultdict(int)
    for name, m in image_mask.items():
        k = source_key(name)
        if k:
            masks[k] |= m
    out: dict[str, Any] = {}
    for kind in KINDS:
        groups = {k: m for k, m in masks.items() if key_kind(k) == kind}
        classes = Counter(split_class(m) for m in groups.values())
        out[kind] = {
            "groups": len(groups),
            "split_class": dict(sorted(classes.items())),
            "mixed_examples": sorted(k for k, m in groups.items() if m == 3)[:_EXAMPLES],
        }
    return out


def with_train_source_in_csv(
    images: Iterable[str], image_mask: Mapping[str, int]
) -> dict[str, int]:
    """Per kind: images whose source group has a ``train`` row anywhere in the CSV.

    Unlike the on-disk comparison this covers the whole CSV, i.e. what a model trained on
    the CSV ``train`` split could have seen.
    """
    train_keys = {k for n, m in image_mask.items() if m & 1 and (k := source_key(n))}
    hits = Counter(key_kind(k) for n in set(images) if (k := source_key(n)) in train_keys)
    return dict(sorted(hits.items()))


def _sharing(train: set[str], golden: set[str], main_ids: set[str]) -> dict[str, Any]:
    shared = sharing_source(train, golden)
    keys = Counter(key_kind(k) for k in {source_key(n) for n in shared} if k)
    return {
        "train_images": len(shared),
        "train_images_by_kind": dict(
            sorted(Counter(key_kind(source_key(n) or "") for n in shared).items())
        ),
        "distinct_keys_by_kind": dict(sorted(keys.items())),
        "train_images_in_main_corpus": sum(1 for n in shared if n in main_ids),
        "examples": shared[:_EXAMPLES],
    }


def source_overlap_report(
    image_mask: Mapping[str, int],
    disk: Iterable[str],
    subset: Mapping[str, str],
    golden_candidates: Iterable[str],
    golden: Iterable[str],
    main_ids: Iterable[str],
) -> dict[str, Any]:
    """Source keys of golden vs train-only on-disk images, and the CSV split per source.

    Train candidates are on-disk images whose CSV split class is ``train_only``, minus the
    golden candidates (they are val-only, so the difference is a no-op guard).
    """
    disk_set, cand, gold, main = set(disk), set(golden_candidates), set(golden), set(main_ids)
    train = {
        n for n in disk_set if n in image_mask and split_class(image_mask[n]) == "train_only"
    } - cand
    by_subset: dict[str, Counter[str]] = defaultdict(Counter)
    for n in disk_set:
        k = source_key(n)
        by_subset[subset.get(n, "not_in_csv")][key_kind(k) if k else "none"] += 1
    return {
        "csv": {
            "unique_images": len(image_mask),
            **_by_kind(image_mask),
            "split_by_source": csv_split_by_source(image_mask),
        },
        "disk": {
            **_by_kind(disk_set),
            "kind_by_subset": {s: dict(sorted(c.items())) for s, c in sorted(by_subset.items())},
        },
        "golden_candidates": _by_kind(cand),
        "golden": _by_kind(gold),
        "train_candidates": {**_by_kind(train), "group_sizes": _group_sizes(train)},
        "train_sharing_golden_candidates": _sharing(train, cand, main),
        "train_sharing_golden": _sharing(train, gold, main),
        "main_corpus_sharing_golden": len(sharing_source(main, gold)),
        "golden_with_train_source_in_csv": with_train_source_in_csv(gold, image_mask),
    }
