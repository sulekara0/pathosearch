"""Golden-set evaluation: run systems on a split, score against qrels, write a report.

Systems are named ``<encoder>`` or ``<encoder>+rerank`` (e.g. ``quilt+rerank``). They run
through ``pipeline.run_text_search`` and ``reranker.rerank_results`` with the app's
alpha/beta/pool settings. The domain gate is **off** during evaluation. Encoders and
indexes are injected as ``runners`` so tests use fake indexes.

Reported per threshold (strict: grade >= 2, lenient: grade >= 1): P@k, Hit@k, Recall@k,
MRR and graded nDCG@k, with seeded paired-bootstrap 95% CIs over queries (the same
resampled queries for every system) and CIs of the difference to the first system.

Guards: the test split runs only with an explicit flag and the report is stamped TEST;
parameters (alpha/beta, pool) are tuned on dev only. Circularity checks: all systems are
also scored on pairs whose annotator never opened the caption, and the Spearman
correlation between grade and caption-overlap is reported.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from pathosearch.config import require_file
from pathosearch.eval import metrics as M
from pathosearch.eval.golden import (
    CorpusEntry,
    GoldenPaths,
    GoldenQuery,
    Judgment,
    Qrels,
    latest_judgments,
    load_judgments,
    load_queries,
    read_trec_qrels,
)
from pathosearch.eval.labeling import load_first_captions
from pathosearch.eval.splits import SPLIT_NAMES, load_splits
from pathosearch.pipeline import (
    RERANK_ALPHA,
    RERANK_BETA,
    RERANK_POOL_FACTOR,
    SearchIndex,
    TextEncoder,
    run_text_search,
)
from pathosearch.reranker import caption_overlap, rerank_results

ENCODERS = ("clip", "quilt")
THRESHOLDS = {"strict": 2, "lenient": 1}
METRIC_NAMES = ("p", "hit", "recall", "mrr", "ndcg")
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42
CI_LEVEL = 0.95
# Fixed before looking at any data: |rho| above this flags label/caption dependence.
SPEARMAN_WARN = 0.5
EXCLUDED_WARN = 0.30  # max share of judged pairs removed before warning
REFERENCE_ANNOTATOR = "A"
CAVEATS = (
    "QuiltNet held-out status of CSV val split: unverified (STATUS §7.8.3).",
    "Judgments are pooled: unjudged images count as non-relevant (see the random control "
    "sample of the pool for how often this is wrong).",
    "Annotators are not pathologists; grades are noisy (see the agreement report).",
)

Runner = Callable[[str], list[str]]  # query text -> ranked image ids
Table = dict[str, dict[str, dict[str, dict[str, float]]]]  # thr -> system -> metric -> qid -> value


# --- systems -----------------------------------------------------------------------


@dataclass(frozen=True)
class SystemSpec:
    name: str
    encoder: str
    rerank: bool


def parse_system(name: str) -> SystemSpec:
    encoder, plus, suffix = name.partition("+")
    if encoder not in ENCODERS or (plus and suffix != "rerank"):
        raise ValueError(f"unknown system {name!r}; use <{'|'.join(ENCODERS)}>[+rerank]")
    return SystemSpec(name, encoder, bool(plus))


def make_runner(encoder: TextEncoder, idx: SearchIndex, rerank: bool, k: int) -> Runner:
    """Same retrieval path as the app (gate off): search, then optional lexical rerank."""

    def run(query: str) -> list[str]:
        fetch_k = min(k * RERANK_POOL_FACTOR if rerank else k, idx.index.ntotal)
        results, _, _ = run_text_search(encoder, idx, query, fetch_k)
        if rerank:
            results = rerank_results(query, results, topk=k, alpha=RERANK_ALPHA, beta=RERANK_BETA)
        return [r["id"] for r in results[:k]]

    return run


def golden_caption_map(corpus: Sequence[CorpusEntry], csv_path: Path) -> dict[str, dict]:
    """``{image: {"caption", "pathology"}}`` for the golden corpus (first CSV caption)."""
    caps = load_first_captions(csv_path, [e.image for e in corpus])
    return {e.image: {"caption": caps.get(e.image, ""), "pathology": e.pathology} for e in corpus}


# --- guards ------------------------------------------------------------------------


def check_split_allowed(split: str, i_know_this_is_test: bool) -> None:
    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}, got {split!r}")
    if split == "test" and not i_know_this_is_test:
        raise ValueError(
            "the test split is held out for one final run; pass --i-know-this-is-test "
            "to confirm (tune alpha/beta and pool size on dev only)"
        )


# --- scoring -----------------------------------------------------------------------


def per_query_metrics(
    ranked: Sequence[str], qrels: Mapping[str, int], k: int, min_rel: int
) -> dict:
    return {
        "p": M.precision_at_k_qrels(ranked, qrels, k, min_rel),
        "hit": M.hit_at_k(ranked, qrels, k, min_rel),
        "recall": M.recall_at_k(ranked, qrels, k, min_rel),
        "mrr": M.reciprocal_rank(ranked, qrels, k, min_rel),
        "ndcg": M.graded_ndcg_at_k(ranked, qrels, k, min_rel),
    }


def score_runs(
    runs: Mapping[str, Mapping[str, Sequence[str]]],
    qrels: Qrels,
    k: int,
    excluded: Mapping[str, set[str]] | None = None,
) -> Table:
    """Per-query metric values; NaN where a query has no relevant judged image.

    ``excluded`` images are removed from the ranked lists before scoring (condensed lists),
    so they count neither as relevant nor as non-relevant. A list that shrinks is scored at
    its remaining length (``min(k, len)``); a query left with no results gets NaN.
    """
    table: Table = {}
    for thr, min_rel in THRESHOLDS.items():
        table[thr] = {}
        for system, per_query in runs.items():
            cols: dict[str, dict[str, float]] = {m: {} for m in METRIC_NAMES}
            for qid, ranked in per_query.items():
                if excluded:
                    ranked = [d for d in ranked if d not in excluded.get(qid, ())]
                k_q = min(k, len(ranked))
                if k_q == 0:
                    vals = {m: math.nan for m in METRIC_NAMES}
                else:
                    vals = per_query_metrics(ranked, qrels.get(qid, {}), k_q, min_rel)
                for m, v in vals.items():
                    cols[m][qid] = v
            table[thr][system] = cols
    return table


def _num(x: float) -> float | None:
    return None if math.isnan(x) else round(float(x), 4)


def summarize(
    table: Table,
    systems: Sequence[str],
    categories: Mapping[str, str],
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Means, bootstrap CIs, differences to ``systems[0]`` and a per-category breakdown."""
    rng = np.random.default_rng(seed)
    lo_q, hi_q = (1 - CI_LEVEL) / 2 * 100, (1 + CI_LEVEL) / 2 * 100
    out: dict[str, Any] = {}
    for thr, per_system in table.items():
        out[thr] = {"systems": {s: {} for s in systems}, "n_queries_without_relevant": {}}
        for metric in METRIC_NAMES:
            qids = sorted(per_system[systems[0]][metric])
            mat = np.array([[per_system[s][metric][q] for q in qids] for s in systems])
            keep = ~np.isnan(mat).any(axis=0)  # NaN depends on qrels only, so equal for all
            out[thr]["n_queries_without_relevant"][metric] = int((~keep).sum())
            mat = mat[:, keep]
            n = mat.shape[1]
            if n == 0:
                for s in systems:
                    out[thr]["systems"][s][metric] = {"mean": None, "ci": None, "n": 0}
                continue
            idx = rng.integers(0, n, size=(n_boot, n))
            boot = mat[:, idx].mean(axis=2)  # (systems, n_boot), same resamples for all systems
            for i, s in enumerate(systems):
                entry: dict[str, Any] = {
                    "mean": _num(mat[i].mean()),
                    "ci": [_num(np.percentile(boot[i], lo_q)), _num(np.percentile(boot[i], hi_q))],
                    "n": n,
                }
                if i > 0:
                    d = boot[i] - boot[0]
                    dlo, dhi = np.percentile(d, lo_q), np.percentile(d, hi_q)
                    entry["diff_vs_first"] = {
                        "mean": _num(mat[i].mean() - mat[0].mean()),
                        "ci": [_num(dlo), _num(dhi)],
                        "excludes_zero": bool(dlo > 0 or dhi < 0),
                    }
                out[thr]["systems"][s][metric] = entry
        out[thr]["by_category"] = _by_category(per_system, systems, categories)
    return out


def _by_category(
    per_system: Mapping, systems: Sequence[str], categories: Mapping[str, str]
) -> dict:
    result: dict[str, Any] = {}
    for cat in sorted(set(categories.values())):
        result[cat] = {}
        for s in systems:
            row: dict[str, Any] = {}
            for metric in METRIC_NAMES:
                vals = [
                    v
                    for q, v in per_system[s][metric].items()
                    if categories.get(q) == cat and not math.isnan(v)
                ]
                row[metric] = _num(float(np.mean(vals))) if vals else None
            row["n_queries"] = len(
                [
                    q
                    for q, v in per_system[s]["p"].items()
                    if categories.get(q) == cat and not math.isnan(v)
                ]
            )
            result[cat][s] = row
    return result


# --- circularity checks --------------------------------------------------------------


def caption_seen_exclusions(
    qrels: Qrels, judgments: Sequence[Judgment]
) -> tuple[Qrels, dict[str, set[str]]]:
    """Split qrels into pairs kept for the circularity check and pairs to exclude.

    A pair is excluded when the reference annotator opened its caption, or when there is
    no reference judgment (caption exposure unknown). Excluded pairs are removed from the
    qrels *and* from the ranked lists (see :func:`score_runs`): they are treated as if
    never judged, not as non-relevant.
    """
    latest = latest_judgments(judgments)
    kept: Qrels = {}
    excluded: dict[str, set[str]] = {}
    for qid, per_image in qrels.items():
        for image, grade in per_image.items():
            j = latest.get((qid, image))
            if j is not None and not j.caption_seen:
                kept.setdefault(qid, {})[image] = grade
            else:
                excluded.setdefault(qid, set()).add(image)
    return kept, excluded


def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    for v in np.unique(x):  # average ranks over ties
        mask = x == v
        ranks[mask] = ranks[mask].mean()
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman rank correlation; ``None`` if undefined (n < 3 or a constant input)."""
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(a) != len(b):
        raise ValueError("x and y must have the same length")
    if len(a) < 3:
        return None
    ra, rb = _ranks(a), _ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def label_caption_correlation(
    qrels: Qrels, queries: Mapping[str, GoldenQuery], captions: Mapping[str, str]
) -> dict[str, Any]:
    """Spearman(grade, weighted Jaccard(query, caption)) over all judged pairs."""
    grades: list[float] = []
    overlaps: list[float] = []
    for qid, per_image in qrels.items():
        for image, grade in per_image.items():
            grades.append(float(grade))
            overlaps.append(caption_overlap(queries[qid].text, captions.get(image, "")))
    rho = spearman(grades, overlaps)
    return {
        "spearman_rho": None if rho is None else round(rho, 4),
        "n_pairs": len(grades),
        "warn_threshold": SPEARMAN_WARN,
        "warning": (
            "labels depend on caption wording; lexical rerank gains may be inflated"
            if rho is not None and abs(rho) >= SPEARMAN_WARN
            else None
        ),
    }


# --- run + report --------------------------------------------------------------------


def package_versions(
    names: Sequence[str] = (
        "numpy",
        "faiss-cpu",
        "torch",
        "open_clip_torch",
        "transformers",
        "pathosearch",
    ),
) -> dict[str, str]:
    out: dict[str, str] = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = "not installed"
    return out


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


def run_eval(
    golden: GoldenPaths,
    split: str,
    systems: Sequence[str],
    k: int,
    runners: Mapping[str, Runner],
    captions: Mapping[str, str],
    command: str,
    i_know_this_is_test: bool = False,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
    sha: str | None = None,
    packages: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Evaluate ``systems`` on ``split`` of the golden set and return the report dict."""
    check_split_allowed(split, i_know_this_is_test)
    if not systems or len(set(systems)) != len(systems):
        raise ValueError("systems must be a non-empty list without duplicates")
    specs = [parse_system(s) for s in systems]
    missing = [s.name for s in specs if s.name not in runners]
    if missing:
        raise ValueError(f"no runner provided for {missing}")

    queries = {q.qid: q for q in load_queries(golden.queries)}
    qids = load_splits(golden.splits)[split]
    unknown = [q for q in qids if q not in queries]
    if unknown:
        raise ValueError(f"splits.json refers to unknown qids {unknown[:5]}")
    qrels = {q: g for q, g in read_trec_qrels(golden.qrels).items() if q in set(qids)}
    reference = load_judgments(require_file(golden.judgments(REFERENCE_ANNOTATOR)))
    unsure = sum(
        1 for j in latest_judgments(reference).values() if j.grade is None and j.qid in set(qids)
    )

    runs = {s.name: {q: runners[s.name](queries[q].text) for q in qids} for s in specs}
    names = [s.name for s in specs]
    categories = {q: queries[q].category for q in qids}
    unseen, excluded = caption_seen_exclusions(qrels, reference)
    n_total = sum(len(v) for v in qrels.values())
    n_excluded = sum(len(v) for v in excluded.values())
    share = n_excluded / n_total if n_total else 0.0
    unseen_summary = summarize(
        score_runs(runs, unseen, k, excluded), names, categories, n_boot, seed
    )

    return {
        "golden": golden.root.name,
        "split": split,
        "split_stamp": "TEST" if split == "test" else "dev",
        "k": k,
        "systems": names,
        "first_system_is_reference_for_differences": names[0],
        "command": command,
        "config": {
            "alpha": RERANK_ALPHA,
            "beta": RERANK_BETA,
            "rerank_pool_factor": RERANK_POOL_FACTOR,
            "domain_gate": "off",
            "thresholds": THRESHOLDS,
            "bootstrap": {"n": n_boot, "seed": seed, "ci": CI_LEVEL, "paired": True},
        },
        "git_sha": sha if sha is not None else git_sha(),
        "packages": dict(packages) if packages is not None else package_versions(),
        "n_queries": len(qids),
        "unsure_pairs": unsure,
        "metrics": summarize(score_runs(runs, qrels, k), names, categories, n_boot, seed),
        "metrics_caption_unseen": {
            "note": "pairs whose caption the reference annotator opened (or whose exposure is "
            "unknown) are removed from qrels and from ranked lists, i.e. treated as unjudged",
            "pairs_total": n_total,
            "pairs_removed": n_excluded,
            "share_removed": round(share, 4),
            "warning": (
                f"{share:.0%} of judged pairs removed (> {EXCLUDED_WARN:.0%}); "
                "the caption-unseen estimate rests on a small, possibly biased subset"
                if share > EXCLUDED_WARN
                else None
            ),
            "queries_total": len(qids),
            "queries_with_relevant_remaining": {
                thr: len(qids) - unseen_summary[thr]["n_queries_without_relevant"]["ndcg"]
                for thr in THRESHOLDS
            },
            **unseen_summary,
        },
        "circularity": label_caption_correlation(qrels, queries, captions),
        "caveats": list(CAVEATS),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    stamp = " — **TEST SPLIT**" if report["split"] == "test" else ""
    lines = [
        f"# Eval: {report['golden']} / {report['split']}{stamp}",
        "",
        f"- command: `{report['command']}`",
        f"- git: `{report['git_sha']}` · k={report['k']} · queries={report['n_queries']} · "
        f"unsure pairs={report['unsure_pairs']}",
        f"- config: {json.dumps(report['config'])}",
        "",
    ]

    def table(block: Mapping[str, Any], title: str) -> None:
        for thr in THRESHOLDS:
            lines.extend([f"## {title} — {thr} (grade >= {THRESHOLDS[thr]})", ""])
            lines.extend(["| system | " + " | ".join(METRIC_NAMES) + " |"])
            lines.extend(["|---|" + "---|" * len(METRIC_NAMES)])
            for s, per_metric in block[thr]["systems"].items():
                cells = []
                for m in METRIC_NAMES:
                    e = per_metric[m]
                    cells.append(
                        "n/a"
                        if e["mean"] is None
                        else f"{e['mean']:.3f} [{e['ci'][0]:.3f}, {e['ci'][1]:.3f}]"
                    )
                lines.append(f"| {s} | " + " | ".join(cells) + " |")
            lines.append("")

    table(report["metrics"], "Metrics")
    table(report["metrics_caption_unseen"], "Metrics, caption-unseen pairs only")
    cu = report["metrics_caption_unseen"]
    lines.extend(
        [
            f"Caption-unseen: {cu['pairs_removed']} of {cu['pairs_total']} judged pairs removed "
            f"({cu['share_removed']:.1%}); queries with a relevant image remaining: "
            f"{cu['queries_with_relevant_remaining']} of {cu['queries_total']}.",
            "",
        ]
    )
    if cu["warning"]:
        lines.extend([f"**Warning:** {cu['warning']}", ""])
    circ = report["circularity"]
    lines += [
        "## Circularity check",
        "",
        f"Spearman(grade, caption overlap) = {circ['spearman_rho']} over {circ['n_pairs']} pairs.",
    ]
    if circ["warning"]:
        lines.append(f"**Warning:** {circ['warning']}")
    lines += ["", "## Caveats", ""] + [f"- {c}" for c in report["caveats"]] + [""]
    return "\n".join(lines)


def write_report(report: Mapping[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"eval_{report['golden']}_{report['split']}_{report['git_sha']}"
    json_path, md_path = out_dir / f"{stem}.json", out_dir / f"{stem}.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    md_path.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    return json_path, md_path


def command_line() -> str:
    return " ".join(["uv run python", *sys.argv])
