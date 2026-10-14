"""Pathology label normalization.

Quilt-1M metadata spells the same category differently (``unknown`` vs
``Unknown``). Every place that creates or compares a pathology label goes
through :func:`normalize_pathology`, so those spellings count as one category.
"""

from __future__ import annotations

from collections.abc import Iterable


def normalize_pathology(value: object) -> str:
    """Return ``value`` as a stripped, lower-case label; ``None`` -> ``""``."""
    if value is None:
        return ""
    return str(value).strip().lower()


def normalize_labels(values: Iterable[object]) -> set[str]:
    """Normalize several labels into a set (e.g. a query's expected categories)."""
    return {normalize_pathology(v) for v in values}
