"""Descriptive analysis of near-duplicate cosines (golden vs main corpus).

Measures only; it never changes the leakage threshold. See scripts/leakage_analysis.py.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

_CHUNK = 256
THRESHOLDS = (0.95, 0.99, 0.999, 0.9999)
PERCENTILES = (50, 90, 99)


def _unit(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def nearest_cosine(
    query: np.ndarray, ref: np.ndarray, exclude_self_rows: Sequence[int] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Max cosine and argmax of each query row against ``ref``.

    ``exclude_self_rows[i]`` is the row of ``ref`` that is query ``i`` itself (masked out).
    """
    q, r = _unit(query), _unit(ref)
    if exclude_self_rows is not None and len(exclude_self_rows) != len(q):
        raise ValueError("exclude_self_rows must have one entry per query")
    best = np.empty(len(q), dtype=np.float32)
    arg = np.empty(len(q), dtype=np.int64)
    for s in range(0, len(q), _CHUNK):
        sims = q[s : s + _CHUNK] @ r.T
        if exclude_self_rows is not None:
            for k in range(sims.shape[0]):
                sims[k, exclude_self_rows[s + k]] = -np.inf
        a = sims.argmax(axis=1)
        arg[s : s + _CHUNK] = a
        best[s : s + _CHUNK] = sims[np.arange(len(a)), a]
    return best, arg


def distribution(values: np.ndarray, bins: int = 20) -> dict[str, Any]:
    """Percentiles, max, counts at fixed thresholds and a histogram on [min, 1]."""
    v = np.asarray(values, dtype=np.float64)
    hist, edges = np.histogram(v, bins=bins, range=(float(v.min()), 1.0))
    return {
        "n": int(v.size),
        "min": round(float(v.min()), 4),
        "percentiles": {f"p{p}": round(float(np.percentile(v, p)), 4) for p in PERCENTILES},
        "max": round(float(v.max()), 4),
        "count_ge": {str(t): int((v >= t).sum()) for t in THRESHOLDS},
        "histogram": {
            "edges": [round(float(e), 4) for e in edges],
            "counts": [int(c) for c in hist],
        },
    }


def clusters(pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """Connected components over the images that appear in ``pairs`` (union-find)."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        parent[find(a)] = find(b)
    groups: dict[str, int] = {}
    for x in list(parent):
        root = find(x)
        groups[root] = groups.get(root, 0) + 1
    sizes = sorted(groups.values(), reverse=True)
    return {
        "n_pairs": len(pairs),
        "n_images": len(parent),
        "n_clusters": len(sizes),
        "largest_cluster_sizes": sizes[:10],
        "singleton_pairs_clusters_of_2": sum(1 for s in sizes if s == 2),
    }


def within_pairs(emb: np.ndarray, ids: Sequence[str], tau: float) -> list[tuple[str, str]]:
    """Unordered image pairs (i < j) with cosine > ``tau``."""
    u = _unit(emb)
    out: list[tuple[str, str]] = []
    for s in range(0, len(u), _CHUNK):
        sims = u[s : s + _CHUNK] @ u.T
        for row in range(sims.shape[0]):
            i = s + row
            for j in np.nonzero(sims[row] > tau)[0]:
                if j > i:
                    out.append((ids[i], ids[int(j)]))
    return out
