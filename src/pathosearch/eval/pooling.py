"""Depth-k pooling for the golden set.

For each query the pool is the union of every system's top-``depth`` results, plus
``n_random`` images drawn from the rest of the corpus as a control sample: the share
of relevant images among them estimates how often the "unjudged = non-relevant"
assumption is wrong.

Annotators see items in a shuffled ``order`` and never the systems that retrieved
them (blind judging). ``systems`` is stored only for later analysis.

Randomness is per query (seed + CRC32 of the qid), so adding queries later (pilot ->
full set) leaves the pool, order and overlap of existing queries unchanged.
"""

from __future__ import annotations

import zlib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pathosearch.eval.golden import read_jsonl, write_jsonl

DEFAULT_DEPTH = 10
DEFAULT_N_RANDOM = 5
DEFAULT_OVERLAP_FRAC = 0.2
DEFAULT_SEED = 42
SOURCE_POOLED = "pooled"
SOURCE_RANDOM = "random"
SOURCE_R4_EXTENSION = "r4_extension"

Runs = Mapping[str, Mapping[str, Sequence[str]]]  # system -> qid -> ranked image ids


@dataclass(frozen=True)
class PoolItem:
    qid: str
    image: str
    order: int
    source: str
    systems: tuple[str, ...]


def query_rng(seed: int, qid: str) -> np.random.Generator:
    """Generator that depends only on ``seed`` and ``qid`` (stable across runs/processes)."""
    return np.random.default_rng([seed, zlib.crc32(qid.encode("utf-8"))])


def build_pool(
    runs: Runs,
    corpus_ids: Sequence[str],
    depth: int = DEFAULT_DEPTH,
    n_random: int = DEFAULT_N_RANDOM,
    seed: int = DEFAULT_SEED,
) -> list[PoolItem]:
    """Union of top-``depth`` per system + random control images, shuffled per query."""
    if depth < 1 or n_random < 0:
        raise ValueError("depth must be >= 1 and n_random >= 0")
    if not runs:
        raise ValueError("no system runs to pool")
    corpus = sorted(set(corpus_ids))
    corpus_set = set(corpus)
    qids = sorted({qid for per_query in runs.values() for qid in per_query})
    for system, per_query in runs.items():
        missing = set(qids) - set(per_query)
        if missing:
            raise ValueError(f"system {system!r} has no run for qids {sorted(missing)[:5]}")

    pool: list[PoolItem] = []
    for qid in qids:
        retrieved_by: dict[str, set[str]] = {}
        for system in sorted(runs):
            for image in runs[system][qid][:depth]:
                if image not in corpus_set:
                    raise ValueError(f"{system!r} returned {image!r} for {qid}: not in corpus")
                retrieved_by.setdefault(image, set()).add(system)

        rng = query_rng(seed, qid)
        rest = [i for i in corpus if i not in retrieved_by]
        picked = rng.choice(len(rest), size=min(n_random, len(rest)), replace=False)
        randoms = sorted(rest[i] for i in picked)

        items = [(img, SOURCE_POOLED, tuple(sorted(s))) for img, s in sorted(retrieved_by.items())]
        items += [(img, SOURCE_RANDOM, ()) for img in randoms]
        for order, idx in enumerate(rng.permutation(len(items))):
            image, source, systems = items[idx]
            pool.append(PoolItem(qid, image, order, source, systems))
    return pool


def extend_pool(
    pool: Sequence[PoolItem],
    runs: Runs,
    judged: Mapping[str, set[str]],
    depth: int = DEFAULT_DEPTH,
    source: str = SOURCE_R4_EXTENSION,
    seed: int = DEFAULT_SEED,
) -> list[PoolItem]:
    """New pool items: top-``depth`` images of ``runs`` that are neither pooled nor judged.

    Existing items keep their ``order``; new items of a query are shuffled among themselves
    (seed + CRC32 of the qid, independent of the original draw) and ordered after the
    existing ones, so a re-run with the same inputs gives the same result. Systems are kept
    for later analysis only; annotators never see them.
    """
    if depth < 1:
        raise ValueError("depth must be >= 1")
    present = {(it.qid, it.image) for it in pool}
    next_order: dict[str, int] = {}
    for it in pool:
        next_order[it.qid] = max(next_order.get(it.qid, 0), it.order + 1)
    qids = sorted({qid for per_query in runs.values() for qid in per_query})
    new: list[PoolItem] = []
    for qid in qids:
        retrieved_by: dict[str, set[str]] = {}
        for system in sorted(runs):
            for image in runs[system].get(qid, [])[:depth]:
                if (qid, image) not in present and image not in judged.get(qid, set()):
                    retrieved_by.setdefault(image, set()).add(system)
        images = sorted(retrieved_by)
        rng = query_rng(seed + 2, qid)
        base = next_order.get(qid, 0)
        for offset, idx in enumerate(rng.permutation(len(images))):
            image = images[idx]
            new.append(
                PoolItem(qid, image, base + offset, source, tuple(sorted(retrieved_by[image])))
            )
    return new


def overlap_subset(
    pool: Sequence[PoolItem], frac: float = DEFAULT_OVERLAP_FRAC, seed: int = DEFAULT_SEED
) -> list[PoolItem]:
    """Per-query sample (at least one item per query) for the second annotator."""
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    by_query: dict[str, list[PoolItem]] = {}
    for item in pool:
        by_query.setdefault(item.qid, []).append(item)
    subset: list[PoolItem] = []
    for qid in sorted(by_query):
        items = sorted(by_query[qid], key=lambda it: it.order)
        n = max(1, round(len(items) * frac))
        rng = query_rng(seed + 1, qid)  # independent of the pool's own draw
        picked = sorted(rng.choice(len(items), size=n, replace=False))
        subset += [items[i] for i in picked]
    return subset


def pool_stats(pool: Sequence[PoolItem]) -> dict[str, Any]:
    """Counts for the report: pairs, per-query size, sources, unique contributions."""
    per_query = Counter(it.qid for it in pool)
    sizes = list(per_query.values())
    only_one = Counter(it.systems[0] for it in pool if len(it.systems) == 1)
    return {
        "pairs": len(pool),
        "queries": len(per_query),
        "pairs_per_query": {
            "min": min(sizes, default=0),
            "max": max(sizes, default=0),
            "mean": round(sum(sizes) / len(sizes), 2) if sizes else 0.0,
        },
        "by_source": dict(sorted(Counter(it.source for it in pool).items())),
        "unique_to_one_system": dict(sorted(only_one.items())),
    }


def save_pool(path: Path, items: Iterable[PoolItem]) -> None:
    write_jsonl(path, ({**asdict(it), "systems": list(it.systems)} for it in items))


def load_pool(path: Path) -> list[PoolItem]:
    items = [
        PoolItem(
            qid=row["qid"],
            image=row["image"],
            order=int(row["order"]),
            source=row["source"],
            systems=tuple(row["systems"]),
        )
        for row in read_jsonl(path)
    ]
    keys = Counter((it.qid, it.image) for it in items)
    dupes = [k for k, n in keys.items() if n > 1]
    if dupes:
        raise ValueError(f"duplicate (qid, image) in pool: {dupes[:5]}")
    return items
