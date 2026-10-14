"""Query-level dev/test split and query leakage checks.

The split is over *queries*, stratified by query category. The lookup CSV has no
video/source id (only ``subset``), so a source-level split is not possible with
this data (STATUS §7.8.2). Tuning happens on ``dev`` only; ``test`` is read once.

Dev share: ``dev_frac`` if given, else 0.5 for sets of at most
``SMALL_SET_MAX_QUERIES`` queries (pilot / small sets) and 0.3 otherwise.
``force_dev`` qids (e.g. the pilot queries, already used to refine the rubric and
tool) always land in dev.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np

from pathosearch.config import require_file
from pathosearch.eval.golden import GoldenQuery

DEFAULT_SEED = 42
SMALL_SET_MAX_QUERIES = 50
SMALL_SET_DEV_FRAC = 0.5
LARGE_SET_DEV_FRAC = 0.3
NEAR_DUPLICATE_JACCARD = 0.8
SPLIT_NAMES = ("dev", "test")

Splits = dict[str, list[str]]  # split name -> sorted qids


def default_dev_frac(n_queries: int) -> float:
    return SMALL_SET_DEV_FRAC if n_queries <= SMALL_SET_MAX_QUERIES else LARGE_SET_DEV_FRAC


def _dev_quota(sizes: dict[str, int], target: int) -> dict[str, int]:
    """Spread ``target`` dev slots over categories in proportion to their size.

    Largest-remainder rounding (ties broken by category name) so the total is exact.
    """
    total = sum(sizes.values())
    exact = {c: n * target / total for c, n in sizes.items()}
    quota = {c: math.floor(x) for c, x in exact.items()}
    left = target - sum(quota.values())
    for c in sorted(sizes, key=lambda c: (-(exact[c] - quota[c]), c))[:left]:
        quota[c] += 1
    return quota


def split_queries(
    queries: Sequence[GoldenQuery],
    dev_frac: float | None = None,
    seed: int = DEFAULT_SEED,
    force_dev: Iterable[str] = (),
) -> Splits:
    """Stratified (by category) dev/test split of query ids; deterministic for a seed."""
    if not queries:
        raise ValueError("no queries to split")
    frac = default_dev_frac(len(queries)) if dev_frac is None else dev_frac
    if not 0.0 < frac < 1.0:
        raise ValueError(f"dev_frac must be in (0, 1), got {frac}")
    qids = {q.qid for q in queries}
    forced = set(force_dev)
    unknown = forced - qids
    if unknown:
        raise ValueError(f"force_dev qids not in queries: {sorted(unknown)}")

    by_cat: dict[str, list[str]] = defaultdict(list)
    for q in queries:
        by_cat[q.category].append(q.qid)
    quota = _dev_quota({c: len(v) for c, v in by_cat.items()}, round(len(queries) * frac))

    rng = np.random.default_rng(seed)
    dev: list[str] = []
    for cat in sorted(by_cat):
        members = sorted(by_cat[cat])
        pinned = [q for q in members if q in forced]
        free = [members[i] for i in rng.permutation(len(members)) if members[i] not in forced]
        dev += pinned + free[: max(0, quota[cat] - len(pinned))]
    dev_set = set(dev)
    return {"dev": sorted(dev_set), "test": sorted(qids - dev_set)}


def save_splits(path: Path, splits: Splits, seed: int, dev_frac: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"seed": seed, "dev_frac": dev_frac, **{s: sorted(splits[s]) for s in SPLIT_NAMES}}
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")


def load_splits(path: Path) -> Splits:
    doc = json.loads(require_file(path).read_text(encoding="utf-8"))
    splits = {s: list(doc[s]) for s in SPLIT_NAMES}
    both = set(splits["dev"]) & set(splits["test"])
    if both:
        raise ValueError(f"qids in both dev and test: {sorted(both)[:5]}")
    return splits


# --- leakage ------------------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9]+")


def query_tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def token_jaccard(a: str, b: str) -> float:
    ta, tb = set(query_tokens(a)), set(query_tokens(b))
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / len(ta | tb)


@dataclass
class SplitLeakageReport:
    """Errors make a split unusable; ``near_duplicates`` are warnings to review."""

    in_both_splits: list[str] = field(default_factory=list)
    unassigned: list[str] = field(default_factory=list)
    unknown_qids: list[str] = field(default_factory=list)
    cross_split_exact_duplicates: list[tuple[str, str]] = field(default_factory=list)
    near_duplicates: list[tuple[str, str, float, bool]] = field(default_factory=list)
    """(qid_a, qid_b, token Jaccard, crosses dev/test)"""

    @property
    def errors(self) -> list[str]:
        out = []
        if self.in_both_splits:
            out.append(f"qids in both splits: {self.in_both_splits}")
        if self.unassigned:
            out.append(f"queries in no split: {self.unassigned}")
        if self.unknown_qids:
            out.append(f"split qids not in queries: {self.unknown_qids}")
        if self.cross_split_exact_duplicates:
            out.append(f"same query text in dev and test: {self.cross_split_exact_duplicates}")
        return out


def check_split_leakage(
    queries: Sequence[GoldenQuery],
    splits: Splits,
    near_duplicate_jaccard: float = NEAR_DUPLICATE_JACCARD,
) -> SplitLeakageReport:
    dev, test = set(splits["dev"]), set(splits["test"])
    qids = {q.qid for q in queries}
    report = SplitLeakageReport(
        in_both_splits=sorted(dev & test),
        unassigned=sorted(qids - dev - test),
        unknown_qids=sorted((dev | test) - qids),
    )
    for a, b in combinations(sorted(queries, key=lambda q: q.qid), 2):
        crosses = (a.qid in dev and b.qid in test) or (a.qid in test and b.qid in dev)
        if crosses and query_tokens(a.text) == query_tokens(b.text):
            report.cross_split_exact_duplicates.append((a.qid, b.qid))
        score = token_jaccard(a.text, b.text)
        if score >= near_duplicate_jaccard:
            report.near_duplicates.append((a.qid, b.qid, round(score, 4), crosses))
    return report


def assert_no_split_leakage(queries: Sequence[GoldenQuery], splits: Splits) -> SplitLeakageReport:
    """Raise ValueError on any split error; return the report (with warnings) otherwise."""
    report = check_split_leakage(queries, splits)
    if report.errors:
        raise ValueError("; ".join(report.errors))
    return report
