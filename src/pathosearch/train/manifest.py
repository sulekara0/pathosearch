"""R4 training manifest: which train-only images may be used to fine-tune, and with which captions.

Candidates are on-disk images whose CSV split class is ``train_only``. Exclusion steps run
in a fixed order and every step records how many images it removed (ADR-0004):

    no_embedding     no main-corpus CLIP/QuiltNet embedding, so the near-copy check is impossible
    golden_ids       a golden candidate itself (guard; candidates are val-only)
    golden_nearest   the main-corpus image a golden candidate was excluded for (ADR-0003)
    near_dup_golden  cosine >= tau to any golden candidate in ANY encoder
    source_group     shares a video / tweet / PubMed figure key with a golden candidate
    caption_equal    one of its captions equals (normalised) a golden candidate caption
    no_caption       no usable caption (``not_histology == 0`` and non-empty, as in prepare_quilt)

The internal validation set is cut from the remainder by whole source groups, then
validation images with a near copy in train are dropped. Scale subsets are nested
prefixes of one hash order of the train images. Nothing here uses a random generator:
every order is ``sha1(f"{seed}:{key}")``.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pathosearch.config import require_file
from pathosearch.data.prepare_quilt import parse_pathology
from pathosearch.data.source_groups import group_key, sharing_source, source_key
from pathosearch.eval.dedup import exclude_near_dup
from pathosearch.eval.heldout import assert_disjoint
from pathosearch.eval.leakage_analysis import nearest_cosine

NEAR_DUP_TAU = 0.99  # ADR-0003 golden gate, reused for training exclusion
VAL_SIZE = 2000
SEED = 42
SCALES: dict[str, int | None] = {"10k": 10_000, "30k": 30_000, "full": None}
STEPS = (
    "no_embedding",
    "golden_ids",
    "golden_nearest",
    "near_dup_golden",
    "source_group",
    "caption_equal",
    "no_caption",
)
_EXAMPLES = 5
_CSV_COLS = ["image_path", "caption", "pathology", "not_histology", "subset"]


# --- deterministic order and captions -------------------------------------------------


def stable_hash(seed: int, key: str) -> int:
    return int(hashlib.sha1(f"{seed}:{key}".encode()).hexdigest(), 16)


def normalize_caption(text: str) -> str:
    """Lower case, collapsed whitespace: the form used for caption equality."""
    return " ".join(text.lower().split())


def caption_for(image: str, captions: Sequence[str], epoch: int, seed: int = SEED) -> str:
    """Caption of ``image`` in ``epoch``: rotate through its captions from a hashed offset.

    Every caption is used once per ``len(captions)`` epochs; no random state is involved.
    """
    if not captions:
        raise ValueError(f"{image}: no captions")
    return captions[(stable_hash(seed, image) + epoch) % len(captions)]


@dataclass
class ImageMeta:
    category: str  # first pathology of the first CSV row (same rule as select_heldout)
    subset: str
    captions: list[str] = field(default_factory=list)  # CSV order, unique, filtered


def _usable(not_histology: object, caption: object) -> bool:
    """prepare_quilt filter: ``not_histology == 0`` and a non-empty caption."""
    try:
        is_histology = float(str(not_histology)) == 0
    except ValueError:
        return False
    return is_histology and isinstance(caption, str) and caption.strip() != ""


def collect_meta(rows: Iterable[Mapping[str, Any]]) -> dict[str, ImageMeta]:
    """Fold CSV rows (in file order) into one :class:`ImageMeta` per image."""
    out: dict[str, ImageMeta] = {}
    for r in rows:
        name = str(r["image_path"])
        meta = out.get(name)
        if meta is None:
            meta = out[name] = ImageMeta(parse_pathology(r.get("pathology")), str(r["subset"]))
        cap = r.get("caption")
        if _usable(r.get("not_histology"), cap):
            text = str(cap).strip()
            if text not in meta.captions:
                meta.captions.append(text)
    return out


def load_image_meta(csv_path: Path, images: Iterable[str]) -> dict[str, ImageMeta]:
    """:func:`collect_meta` over the CSV rows of ``images`` (streamed in chunks)."""
    wanted = set(images)
    rows: list[dict[str, Any]] = []
    for chunk in pd.read_csv(
        require_file(csv_path), usecols=_CSV_COLS, dtype=str, chunksize=200_000
    ):
        hit = chunk[chunk["image_path"].isin(wanted)]
        rows.extend(hit.to_dict("records"))
    return collect_meta(rows)


# --- exclusion ------------------------------------------------------------------------


@dataclass
class Exclusion:
    kept: list[str]
    steps: list[dict[str, Any]]


def _record(steps: list[dict[str, Any]], name: str, before: list[str], after: list[str]) -> None:
    removed = sorted(set(before) - set(after))
    steps.append(
        {
            "step": name,
            "removed": len(removed),
            "remaining": len(after),
            "examples": removed[:_EXAMPLES],
        }
    )


def exclude(
    candidates: Iterable[str],
    has_embedding: Iterable[str],
    golden: Sequence[str],
    golden_nearest: Iterable[str],
    train_emb: Mapping[str, np.ndarray],
    train_emb_ids: Sequence[str],
    golden_emb: Mapping[str, np.ndarray],
    meta: Mapping[str, ImageMeta],
    tau: float = NEAR_DUP_TAU,
) -> Exclusion:
    """Apply the steps in :data:`STEPS` order. ``golden`` = all golden candidates.

    ``train_emb[enc]`` rows follow ``train_emb_ids``; ``golden_emb[enc]`` rows follow
    ``golden``. ``meta`` must cover the candidates and the golden images.
    """
    steps: list[dict[str, Any]] = []
    cur = sorted(set(candidates))

    emb_set = set(has_embedding)
    nxt = [n for n in cur if n in emb_set]
    _record(steps, "no_embedding", cur, nxt)
    cur = nxt

    golden_set = set(golden)
    nxt = [n for n in cur if n not in golden_set]
    _record(steps, "golden_ids", cur, nxt)
    cur = nxt

    nearest = set(golden_nearest)
    nxt = [n for n in cur if n not in nearest]
    _record(steps, "golden_nearest", cur, nxt)
    cur = nxt

    pos = {n: i for i, n in enumerate(train_emb_ids)}
    missing = [n for n in cur if n not in pos]
    if missing:
        raise ValueError(f"{len(missing)} candidates lack embeddings, e.g. {missing[:3]}")
    rows = [pos[n] for n in cur]
    hits = exclude_near_dup(
        {enc: e[rows] for enc, e in train_emb.items()}, cur, golden_emb, list(golden), tau
    )
    near = {h["image"] for h in hits}
    nxt = [n for n in cur if n not in near]
    _record(steps, "near_dup_golden", cur, nxt)
    cur = nxt

    shared = set(sharing_source(cur, golden))
    nxt = [n for n in cur if n not in shared]
    _record(steps, "source_group", cur, nxt)
    cur = nxt

    golden_caps = {normalize_caption(c) for g in golden for c in meta[g].captions}
    nxt = [n for n in cur if not any(normalize_caption(c) in golden_caps for c in meta[n].captions)]
    _record(steps, "caption_equal", cur, nxt)
    cur = nxt

    nxt = [n for n in cur if meta[n].captions]
    _record(steps, "no_caption", cur, nxt)
    cur = nxt

    assert_disjoint(cur, golden)
    return Exclusion(cur, steps)


# --- validation split and scales --------------------------------------------------------


@dataclass
class Split:
    train: list[str]  # in scale order (scale_rank = index)
    val: list[str]  # sorted
    val_groups: int
    val_dropped_near_dup: list[str]


def split_val_by_group(
    images: Sequence[str], val_size: int = VAL_SIZE, seed: int = SEED
) -> tuple[list[str], list[str], int]:
    """Whole source groups, in hash order, until ``val_size`` images. Returns (train, val, groups)."""
    groups: dict[str, list[str]] = {}
    for n in images:
        groups.setdefault(group_key(n), []).append(n)
    val: list[str] = []
    taken = 0
    for key in sorted(groups, key=lambda k: (stable_hash(seed, k), k)):
        if len(val) >= val_size:
            break
        val.extend(groups[key])
        taken += 1
    val_set = set(val)
    return [n for n in images if n not in val_set], sorted(val), taken


def drop_val_near_train(
    val: Sequence[str],
    train: Sequence[str],
    emb: Mapping[str, np.ndarray],
    emb_ids: Sequence[str],
    tau: float = NEAR_DUP_TAU,
) -> tuple[list[str], list[str]]:
    """Drop validation images with cosine >= ``tau`` to a train image in ANY encoder."""
    if not val or not train:
        return list(val), []
    pos = {n: i for i, n in enumerate(emb_ids)}
    vr, tr = [pos[n] for n in val], [pos[n] for n in train]
    near: set[str] = set()
    for e in emb.values():
        best, _ = nearest_cosine(e[vr], e[tr])
        near.update(val[i] for i in map(int, np.nonzero(best >= tau)[0]))
    return [n for n in val if n not in near], sorted(near)


def scale_order(train: Iterable[str], seed: int = SEED) -> list[str]:
    return sorted(train, key=lambda n: (stable_hash(seed, n), n))


def scale_sizes(n_train: int, scales: Mapping[str, int | None] = SCALES) -> dict[str, int]:
    """Images per scale; ``None`` = all. Raises if a fixed scale does not fit."""
    out: dict[str, int] = {}
    for name, size in scales.items():
        if size is not None and size > n_train:
            raise ValueError(f"scale {name}={size} exceeds {n_train} training images")
        out[name] = n_train if size is None else size
    return out


def make_split(
    kept: Sequence[str],
    emb: Mapping[str, np.ndarray],
    emb_ids: Sequence[str],
    val_size: int = VAL_SIZE,
    seed: int = SEED,
    tau: float = NEAR_DUP_TAU,
) -> Split:
    train, val, groups = split_val_by_group(kept, val_size, seed)
    val, dropped = drop_val_near_train(val, train, emb, emb_ids, tau)
    return Split(scale_order(train, seed), val, groups, dropped)


# --- manifest rows ----------------------------------------------------------------------


def train_rows(split: Split, meta: Mapping[str, ImageMeta]) -> list[dict[str, Any]]:
    return [
        {
            "image": n,
            "scale_rank": i,
            "category": meta[n].category,
            "subset": meta[n].subset,
            "n_captions": len(meta[n].captions),
            "source": source_key(n),
        }
        for i, n in enumerate(split.train)
    ]


def val_rows(split: Split, meta: Mapping[str, ImageMeta]) -> list[dict[str, Any]]:
    return [
        {
            "image": n,
            "category": meta[n].category,
            "subset": meta[n].subset,
            "n_captions": len(meta[n].captions),
            "source": source_key(n),
        }
        for n in split.val
    ]


def summary(
    excl: Exclusion,
    split: Split,
    meta: Mapping[str, ImageMeta],
    n_candidates: int,
    scales: Mapping[str, int | None] = SCALES,
) -> dict[str, Any]:
    """Counts for the meta file: steps, val, scales, categories and captions per image."""
    sizes = scale_sizes(len(split.train), scales)

    def dist(names: Iterable[str]) -> dict[str, int]:
        return dict(sorted(Counter(meta[n].category for n in names).items()))

    return {
        "candidates": n_candidates,
        "exclusion_steps": excl.steps,
        "after_exclusion": len(excl.kept),
        "val": {
            "images": len(split.val),
            "groups": split.val_groups,
            "dropped_near_dup_train": len(split.val_dropped_near_dup),
            "dropped_examples": split.val_dropped_near_dup[:_EXAMPLES],
            "categories": dist(split.val),
        },
        "train": len(split.train),
        "scales": sizes,
        "categories_by_scale": {s: dist(split.train[:k]) for s, k in sizes.items()},
        "captions_per_image": dict(
            sorted(Counter(len(meta[n].captions) for n in split.train).items())
        ),
    }


def render_markdown(meta: Mapping[str, Any]) -> str:
    """Short report of a manifest meta dict (exclusion table, val, scales)."""
    lines = [
        f"# Training manifest: {meta['name']}",
        "",
        f"- command: `{meta['command']}`",
        f"- git: `{meta['git_sha']}` · seed {meta['seed']} · tau {meta['near_dup_tau']}"
        f" · val target {meta['val_size']}",
        "",
        "## Exclusion",
        "",
        "| step | removed | remaining |",
        "|---|---|---|",
        f"| candidates (on disk, train-only) | | {meta['candidates']} |",
    ]
    lines += [
        f"| {s['step']} | {s['removed']} | {s['remaining']} |" for s in meta["exclusion_steps"]
    ]
    v = meta["val"]
    lines += [
        "",
        "## Validation and scales",
        "",
        f"- val: {v['images']} images from {v['groups']} whole source groups; "
        f"{v['dropped_near_dup_train']} dropped for a near copy in train",
        f"- train: {meta['train']} images; scales: "
        + ", ".join(f"{k}={n}" for k, n in meta["scales"].items()),
        "",
    ]
    return "\n".join(lines)
