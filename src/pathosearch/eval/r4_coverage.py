"""Pool coverage of adapter systems, without computing any retrieval metric (ADR-0004 §8).

Ranked lists are produced for the requested splits and only compared against the set of
already judged (query, image) pairs: judged@k and the unjudged pairs per split. No qrels
grade is read and no P/Recall/nDCG is computed, so running this on the test split does not
reveal anything about system quality.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

from pathosearch.config import require_file
from pathosearch.eval.golden import GoldenPaths, load_judgments, load_queries, read_trec_qrels
from pathosearch.eval.run_eval import REFERENCE_ANNOTATOR, Runner, judged_pairs
from pathosearch.eval.splits import load_splits

Runs = Mapping[str, Mapping[str, Sequence[str]]]  # system -> qid -> ranked image ids


def ranked_runs(
    runners: Mapping[str, Runner], golden: GoldenPaths, qids: Collection[str]
) -> dict[str, dict[str, list[str]]]:
    """Ranked list per system and query (queries in ``queries.jsonl`` order)."""
    texts = {q.qid: q.text for q in load_queries(golden.queries)}
    unknown = sorted(set(qids) - set(texts))
    if unknown:
        raise ValueError(f"unknown qids {unknown[:5]}")
    ordered = [q for q in texts if q in set(qids)]
    return {name: {q: run(texts[q]) for q in ordered} for name, run in runners.items()}


def judged_for_split(golden: GoldenPaths, qids: Collection[str]) -> dict[str, set[str]]:
    """Judged pairs of ``qids``, defined exactly as in ``run_eval`` (qrels + A's unsure)."""
    qrels = read_trec_qrels(golden.qrels)
    reference = load_judgments(require_file(golden.judgments(REFERENCE_ANNOTATOR)))
    return judged_pairs(qrels, reference, qids)


def unjudged_pairs(
    runs: Runs, judged: Mapping[str, set[str]], k: int
) -> dict[str, dict[str, list[str]]]:
    """Top-``k`` images per system and query that nobody judged yet."""
    return {
        system: {
            q: [d for d in ranked[:k] if d not in judged.get(q, set())]
            for q, ranked in per_query.items()
        }
        for system, per_query in runs.items()
    }


def coverage_summary(runs: Runs, judged: Mapping[str, set[str]], k: int) -> dict[str, Any]:
    """judged@k and unjudged-pair counts per system, plus the union across systems."""
    missing = unjudged_pairs(runs, judged, k)
    per_system: dict[str, Any] = {}
    union: set[tuple[str, str]] = set()
    for system, per_query in runs.items():
        n_pairs = sum(min(k, len(r)) for r in per_query.values())
        n_unjudged = sum(len(v) for v in missing[system].values())
        per_system[system] = {
            "judged_at_k": round(1 - n_unjudged / n_pairs, 4) if n_pairs else None,
            "pairs": n_pairs,
            "unjudged_pairs": n_unjudged,
            "unjudged_by_query": {q: len(v) for q, v in missing[system].items()},
        }
        union |= {(q, d) for q, v in missing[system].items() for d in v}
    return {"systems": per_system, "unjudged_union": len(union)}


def coverage_report(
    runs_by_split: Mapping[str, Runs],
    judged_by_split: Mapping[str, Mapping[str, set[str]]],
    k: int,
) -> dict[str, Any]:
    return {
        "k": k,
        "splits": {
            split: coverage_summary(runs, judged_by_split[split], k)
            for split, runs in runs_by_split.items()
        },
    }


def split_runs(
    runners: Mapping[str, Runner], golden: GoldenPaths, splits: Sequence[str]
) -> tuple[dict[str, dict[str, dict[str, list[str]]]], dict[str, dict[str, set[str]]]]:
    """Ranked runs and judged pairs per requested split."""
    qids_by_split = load_splits(golden.splits)
    runs = {s: ranked_runs(runners, golden, qids_by_split[s]) for s in splits}
    judged = {s: judged_for_split(golden, qids_by_split[s]) for s in splits}
    return runs, judged


def adapter_dirs(items: Sequence[str]) -> dict[str, Path]:
    """Parse repeated ``NAME=DIR`` arguments."""
    out: dict[str, Path] = {}
    for item in items:
        name, found, path = item.partition("=")
        if not found or not name or not path:
            raise SystemExit(f"expected NAME=DIR, got {item!r}")
        out[name] = Path(path)
    return out
