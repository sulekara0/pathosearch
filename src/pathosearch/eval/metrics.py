"""Ranking metrics.

Two families live here:

1. Category-based (legacy, used by ``compare_models``). A result is relevant iff its
   ``pathology`` label is in the query's ``expected`` categories. Moved verbatim from
   ``compare_models`` (no behaviour change). Known properties, kept as-is and
   documented in tests (see docs/STATUS.md §4):
   - Precision@k divides by ``k`` even when fewer than ``k`` results are returned.
   - nDCG@k uses an ideal DCG that assumes at least ``k`` relevant documents exist,
     so it is pessimistic for categories with few documents.
   - Queries without ``expected`` categories yield NaN (caller aggregates with nanmean).

2. Qrels-based (golden set). Relevance comes from graded human judgments
   ``qrels: {image_id: grade}`` with grades 0/1/2. Binary metrics take ``min_rel``
   (1 = lenient, 2 = strict). Images missing from ``qrels`` count as non-relevant
   (standard pooling assumption). A query with no document at ``grade >= min_rel``
   yields NaN for every metric, so callers can count and exclude it explicitly.
   Ranked lists must not contain duplicate ids (ValueError).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from pathosearch.labels import normalize_labels, normalize_pathology


def is_relevant(result: dict, expected: list[str]) -> bool:
    return bool(expected) and normalize_pathology(result["pathology"]) in normalize_labels(expected)


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


# --- Qrels-based metrics (golden set) ---------------------------------------

Qrels = Mapping[str, int]


def _check_unique(ranked_ids: Sequence[str]) -> None:
    if len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError("ranked_ids contains duplicate ids")


def relevant_ids(qrels: Qrels, min_rel: int = 1) -> set[str]:
    """Ids judged at ``grade >= min_rel``."""
    return {doc for doc, grade in qrels.items() if grade >= min_rel}


def precision_at_k_qrels(
    ranked_ids: Sequence[str], qrels: Qrels, k: int, min_rel: int = 1
) -> float:
    """Relevant hits in the top ``k`` divided by ``k`` (same convention as the legacy P@k)."""
    _check_unique(ranked_ids)
    rel = relevant_ids(qrels, min_rel)
    if not rel:
        return math.nan
    return sum(1 for d in ranked_ids[:k] if d in rel) / k


def hit_at_k(ranked_ids: Sequence[str], qrels: Qrels, k: int, min_rel: int = 1) -> float:
    """1.0 if any relevant id is in the top ``k``, else 0.0."""
    _check_unique(ranked_ids)
    rel = relevant_ids(qrels, min_rel)
    if not rel:
        return math.nan
    return 1.0 if any(d in rel for d in ranked_ids[:k]) else 0.0


def recall_at_k(ranked_ids: Sequence[str], qrels: Qrels, k: int, min_rel: int = 1) -> float:
    """Share of judged-relevant ids found in the top ``k``."""
    _check_unique(ranked_ids)
    rel = relevant_ids(qrels, min_rel)
    if not rel:
        return math.nan
    return sum(1 for d in ranked_ids[:k] if d in rel) / len(rel)


def reciprocal_rank(
    ranked_ids: Sequence[str], qrels: Qrels, k: int | None = None, min_rel: int = 1
) -> float:
    """1 / rank of the first relevant id (within the top ``k`` if given), else 0.0.

    MRR is the mean of this value over queries.
    """
    _check_unique(ranked_ids)
    rel = relevant_ids(qrels, min_rel)
    if not rel:
        return math.nan
    cutoff = ranked_ids if k is None else ranked_ids[:k]
    for rank, d in enumerate(cutoff, start=1):
        if d in rel:
            return 1.0 / rank
    return 0.0


def _gain(grade: int, min_rel: int) -> float:
    return float(2**grade - 1) if grade >= min_rel else 0.0


def graded_ndcg_at_k(ranked_ids: Sequence[str], qrels: Qrels, k: int, min_rel: int = 1) -> float:
    """nDCG@k with gain ``2**grade - 1``; the ideal DCG is built from the actual qrels.

    Unlike the legacy :func:`ndcg_at_k`, a query with few relevant documents can reach 1.0.
    Grades below ``min_rel`` get zero gain.
    """
    _check_unique(ranked_ids)
    if not relevant_ids(qrels, min_rel):
        return math.nan
    dcg = sum(
        _gain(qrels.get(d, 0), min_rel) / math.log2(i + 1)
        for i, d in enumerate(ranked_ids[:k], start=1)
    )
    ideal = sorted((_gain(g, min_rel) for g in qrels.values()), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg
