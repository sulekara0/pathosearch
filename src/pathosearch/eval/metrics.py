"""Ranking metrics with binary, category-based relevance.

A result is relevant iff its ``pathology`` label is in the query's ``expected``
categories. Moved verbatim from ``compare_models`` (no behaviour change).

Known properties, kept as-is and documented in tests (see docs/STATUS.md §4):
- Precision@k divides by ``k`` even when fewer than ``k`` results are returned.
- nDCG@k uses an ideal DCG that assumes at least ``k`` relevant documents exist,
  so it is pessimistic for categories with few documents.
- Queries without ``expected`` categories yield NaN (caller aggregates with nanmean).
"""

from __future__ import annotations

import math


def is_relevant(result: dict, expected: list[str]) -> bool:
    return bool(expected) and result["pathology"] in expected


def precision_at_k(results: list[dict], expected: list[str], k: int) -> float:
    if not expected:
        return math.nan
    hits = sum(1 for r in results[:k] if is_relevant(r, expected))
    return hits / k


def average_precision_at_k(results: list[dict], expected: list[str], k: int) -> float:
    if not expected:
        return math.nan
    n_hits, ap = 0, 0.0
    for i, r in enumerate(results[:k], start=1):
        if is_relevant(r, expected):
            n_hits += 1
            ap += n_hits / i
    return (ap / n_hits) if n_hits > 0 else 0.0


def ndcg_at_k(results: list[dict], expected: list[str], k: int) -> float:
    if not expected:
        return math.nan
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i in range(1, min(k, len(results)) + 1)
        if is_relevant(results[i - 1], expected)
    )
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, k + 1))
    return dcg / idcg if idcg > 0 else 0.0
