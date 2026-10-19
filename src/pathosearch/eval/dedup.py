"""Near-duplicate exclusion and within-set de-duplication for the golden corpus (ADR-0003).

Separate from ``heldout`` on purpose: the leakage *report* keeps the pre-registered
tau=0.95; this module implements the stricter, opt-in *exclusion* step.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from pathosearch.eval.leakage_analysis import nearest_cosine

_CHUNK = 256


def _unit(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def exclude_near_dup(
    golden_emb: Mapping[str, np.ndarray],
    golden_ids: Sequence[str],
    main_emb: Mapping[str, np.ndarray],
    main_ids: Sequence[str],
    tau: float,
) -> list[dict[str, Any]]:
    """Golden images whose max cosine to the main corpus is >= ``tau`` in ANY encoder.

    Returns one record per excluded image: per-encoder max cosine and nearest main image.
    """
    if set(golden_emb) != set(main_emb):
        raise ValueError("golden and main embeddings must cover the same encoders")
    hits: dict[str, dict[str, Any]] = {}
    for enc, g in golden_emb.items():
        if len(g) != len(golden_ids) or len(main_emb[enc]) != len(main_ids):
            raise ValueError(f"{enc}: embeddings and ids differ in length")
        best, arg = nearest_cosine(g, main_emb[enc])
        for i in map(int, np.nonzero(best >= tau)[0]):
            rec = hits.setdefault(golden_ids[i], {"image": golden_ids[i], "encoders": {}})
            rec["encoders"][enc] = {
                "cosine": round(float(best[i]), 6),
                "nearest": main_ids[int(arg[i])],
            }
    return [hits[k] for k in sorted(hits)]


def dedup_within(
    emb: Mapping[str, np.ndarray], ids: Sequence[str], tau: float
) -> tuple[list[str], list[dict[str, Any]]]:
    """Keep one representative per cluster of images linked by cosine >= ``tau`` in ANY encoder.

    Clusters are connected components. The representative is the lexicographically smallest
    image name (deterministic, no seed). Returns (kept ids in input order, dropped records).
    """
    parent = list(range(len(ids)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for enc, e in emb.items():
        if len(e) != len(ids):
            raise ValueError(f"{enc}: embeddings and ids differ in length")
        u = _unit(e)
        for s in range(0, len(u), _CHUNK):
            sims = u[s : s + _CHUNK] @ u.T
            for row in range(sims.shape[0]):
                i = s + row
                for j in np.nonzero(sims[row] >= tau)[0]:
                    if int(j) > i:
                        parent[find(i)] = find(int(j))
    members: dict[int, list[str]] = {}
    for i, name in enumerate(ids):
        members.setdefault(find(i), []).append(name)
    rep = {name: min(group) for group in members.values() for name in group}
    size = {name: len(group) for group in members.values() for name in group}
    kept = [n for n in ids if rep[n] == n]
    dropped = [
        {"image": n, "representative": rep[n], "cluster_size": size[n]}
        for n in sorted(ids)
        if rep[n] != n
    ]
    return kept, dropped
