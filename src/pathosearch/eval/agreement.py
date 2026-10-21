"""Inter-annotator agreement (Cohen's kappa) and adjudication.

Workflow: annotator A judges the whole pool, annotator B the overlap subset. Kappa
is computed on pairs both judged with a grade (``unsure`` on either side is excluded
and counted) *before* adjudication. Disagreements are then resolved into
``adjudication.jsonl``; final qrels = A's latest judgments overridden by the
adjudicated ones.

Weights: ``"none"`` (plain Cohen's kappa) or ``"linear"`` (ordinal 0/1/2: a 0-vs-2
disagreement costs twice a 0-vs-1). Kappa is NaN when undefined (no pairs, or
chance agreement is total, e.g. both annotators used a single grade).

:func:`overlap_agreement` is the detailed report on a fixed overlap subset: per-side
``unsure`` counts, marginals, binary thresholds (lenient >= 1, strict >= 2), seeded
bootstrap CIs (pair-level and query-clustered) and disagreements by category.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from pathosearch.eval.golden import GRADES, Judgment, Qrels, latest_judgments

Weights = Literal["none", "linear"]
Stat = Callable[[Sequence[int], Sequence[int]], float]

BINARY_THRESHOLDS = {"lenient": 1, "strict": 2}
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42
CI_LEVEL = 0.95


def confusion_matrix(
    a: Sequence[int], b: Sequence[int], categories: Sequence[int] = GRADES
) -> np.ndarray:
    """Counts ``m[i, j]`` = items rated ``categories[i]`` by A and ``categories[j]`` by B."""
    if len(a) != len(b):
        raise ValueError("rating sequences differ in length")
    index = {c: i for i, c in enumerate(categories)}
    m = np.zeros((len(categories), len(categories)), dtype=np.int64)
    for x, y in zip(a, b, strict=True):
        if x not in index or y not in index:
            raise ValueError(f"rating outside {tuple(categories)}: ({x}, {y})")
        m[index[x], index[y]] += 1
    return m


def cohen_kappa(
    a: Sequence[int],
    b: Sequence[int],
    categories: Sequence[int] = GRADES,
    weights: Weights = "none",
) -> float:
    """Cohen's kappa ``1 - sum(w*O) / sum(w*E)`` with disagreement weights ``w``."""
    m = confusion_matrix(a, b, categories)
    n = m.sum()
    if n == 0:
        return float("nan")
    k = len(categories)
    i, j = np.indices((k, k))
    if weights == "none":
        w = (i != j).astype(float)
    elif weights == "linear":
        w = np.abs(i - j) / (k - 1)
    else:
        raise ValueError(f"unknown weights {weights!r}")
    observed = m / n
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    denom = float((w * expected).sum())
    if denom == 0.0:
        return float("nan")
    return 1.0 - float((w * observed).sum()) / denom


@dataclass
class AgreementReport:
    pairs: int
    observed_agreement: float
    kappa: float
    kappa_linear: float
    confusion: list[list[int]]
    """rows = annotator A grade, columns = annotator B grade (0, 1, 2)"""
    unsure_excluded: int
    only_a: int
    only_b: int
    disagreements: list[tuple[str, str, int, int]] = field(default_factory=list)
    """(qid, image, grade A, grade B)"""

    def as_dict(self) -> dict[str, Any]:
        return {
            "pairs": self.pairs,
            "observed_agreement": self.observed_agreement,
            "kappa": self.kappa,
            "kappa_linear": self.kappa_linear,
            "confusion_rows_A_cols_B": self.confusion,
            "unsure_excluded": self.unsure_excluded,
            "only_a": self.only_a,
            "only_b": self.only_b,
            "disagreements": [
                {"qid": q, "image": i, "a": ga, "b": gb} for q, i, ga, gb in self.disagreements
            ],
        }


def agreement_report(a: Iterable[Judgment], b: Iterable[Judgment]) -> AgreementReport:
    """Compare two annotators on the (qid, image) pairs both judged (last judgment wins)."""
    la, lb = latest_judgments(a), latest_judgments(b)
    shared = sorted(la.keys() & lb.keys())
    graded = [k for k in shared if la[k].grade is not None and lb[k].grade is not None]
    ga = [int(la[k].grade) for k in graded]  # type: ignore[arg-type]
    gb = [int(lb[k].grade) for k in graded]  # type: ignore[arg-type]
    agree = sum(x == y for x, y in zip(ga, gb, strict=True))
    return AgreementReport(
        pairs=len(graded),
        observed_agreement=agree / len(graded) if graded else float("nan"),
        kappa=cohen_kappa(ga, gb),
        kappa_linear=cohen_kappa(ga, gb, weights="linear"),
        confusion=confusion_matrix(ga, gb).tolist(),
        unsure_excluded=len(shared) - len(graded),
        only_a=len(la.keys() - lb.keys()),
        only_b=len(lb.keys() - la.keys()),
        disagreements=[
            (k[0], k[1], x, y) for k, x, y in zip(graded, ga, gb, strict=True) if x != y
        ],
    )


def unresolved(report: AgreementReport, adjudication: Iterable[Judgment]) -> list[tuple[str, str]]:
    """Disagreeing pairs that have no adjudicated judgment yet."""
    done = latest_judgments(adjudication).keys()
    return [(q, i) for q, i, _, _ in report.disagreements if (q, i) not in done]


def observed_agreement(a: Sequence[int], b: Sequence[int]) -> float:
    """Share of items with identical ratings (NaN if empty)."""
    if len(a) != len(b):
        raise ValueError("rating sequences differ in length")
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a) if a else float("nan")


def binarize(grades: Sequence[int], threshold: int) -> list[int]:
    """1 if ``grade >= threshold`` else 0."""
    return [int(g >= threshold) for g in grades]


def bootstrap_ci(
    a: Sequence[int],
    b: Sequence[int],
    stats: Mapping[str, Stat],
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
    level: float = CI_LEVEL,
    clusters: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Percentile bootstrap CIs; every stat sees the same resamples.

    Without ``clusters`` items are resampled independently. With ``clusters`` (e.g. the
    qid of each pair) whole clusters are resampled, which respects within-query
    correlation. Resamples where a stat is NaN (degenerate) are dropped and counted.
    """
    if len(a) != len(b) or (clusters is not None and len(clusters) != len(a)):
        raise ValueError("rating sequences differ in length")
    rng = np.random.default_rng(seed)
    if clusters is None:
        groups = [np.array([i]) for i in range(len(a))]
    else:
        by: dict[str, list[int]] = {}
        for i, c in enumerate(clusters):
            by.setdefault(c, []).append(i)
        groups = [np.array(v) for _, v in sorted(by.items())]
    aa, bb = np.asarray(a), np.asarray(b)
    values: dict[str, list[float]] = {name: [] for name in stats}
    if groups:
        for _ in range(n_boot):
            pick = rng.integers(0, len(groups), size=len(groups))
            idx = np.concatenate([groups[g] for g in pick])
            ra, rb = aa[idx].tolist(), bb[idx].tolist()
            for name, fn in stats.items():
                values[name].append(fn(ra, rb))
    alpha = (1.0 - level) / 2.0
    out: dict[str, dict[str, Any]] = {}
    for name, vals in values.items():
        finite = np.array([v for v in vals if not np.isnan(v)])
        ci = (
            [float(np.quantile(finite, alpha)), float(np.quantile(finite, 1.0 - alpha))]
            if finite.size
            else [float("nan"), float("nan")]
        )
        out[name] = {"ci": ci, "nan_resamples": len(vals) - int(finite.size)}
    return out


def _agreement_stats(categories: Sequence[int]) -> dict[str, Stat]:
    return {
        "observed_agreement": observed_agreement,
        "kappa": lambda x, y: cohen_kappa(x, y, categories),
        "kappa_linear": lambda x, y: cohen_kappa(x, y, categories, weights="linear"),
    }


def _with_ci(
    a: Sequence[int],
    b: Sequence[int],
    qids: Sequence[str],
    categories: Sequence[int],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    stats = _agreement_stats(categories)
    if len(categories) == 2:
        stats.pop("kappa_linear")  # equals plain kappa with two categories
    pair = bootstrap_ci(a, b, stats, n_boot, seed)
    query = bootstrap_ci(a, b, stats, n_boot, seed, clusters=qids)
    return {
        name: {
            "value": fn(a, b),
            "ci_pair": pair[name]["ci"],
            "ci_query": query[name]["ci"],
            "nan_resamples_pair": pair[name]["nan_resamples"],
            "nan_resamples_query": query[name]["nan_resamples"],
        }
        for name, fn in stats.items()
    }


def _distribution(grades: Iterable[int | None]) -> dict[str, int]:
    out = {str(g): 0 for g in GRADES} | {"unsure": 0}
    for g in grades:
        out["unsure" if g is None else str(g)] += 1
    return out


def overlap_agreement(
    a: Iterable[Judgment],
    b: Iterable[Judgment],
    overlap: Collection[tuple[str, str]],
    query_category: Mapping[str, str],
    query_text: Mapping[str, str] | None = None,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Detailed agreement of A vs B on exactly the ``overlap`` (qid, image) pairs.

    Every overlap pair must have a judgment from both sides (last one wins), otherwise
    ``ValueError``: a partially judged overlap would silently change the sample.
    """
    la, lb = latest_judgments(a), latest_judgments(b)
    keys = sorted(set(overlap))
    for side, latest in (("a", la), ("b", lb)):
        missing = [k for k in keys if k not in latest]
        if missing:
            raise ValueError(f"{len(missing)} overlap pairs not judged by {side}: {missing[:3]}")
    unknown_q = sorted({q for q, _ in keys} - query_category.keys())
    if unknown_q:
        raise ValueError(f"overlap qids without a query category: {unknown_q[:5]}")
    text = query_text or {}

    ua = {k for k in keys if la[k].grade is None}
    ub = {k for k in keys if lb[k].grade is None}
    graded = [k for k in keys if k not in ua | ub]
    ga = [int(la[k].grade) for k in graded]  # type: ignore[arg-type]
    gb = [int(lb[k].grade) for k in graded]  # type: ignore[arg-type]
    qids = [q for q, _ in graded]

    binary = {}
    for name, thr in BINARY_THRESHOLDS.items():
        xa, xb = binarize(ga, thr), binarize(gb, thr)
        binary[name] = {
            "threshold": thr,
            "confusion_rows_a_cols_b": confusion_matrix(xa, xb, (0, 1)).tolist(),
            **_with_ci(xa, xb, qids, (0, 1), n_boot, seed),
        }

    differing = sorted(
        ((k, x, y) for k, x, y in zip(graded, ga, gb, strict=True) if x != y),
        key=lambda t: (-abs(t[1] - t[2]), t[0]),
    )
    disagreements: list[dict[str, Any]] = [
        {
            "qid": k[0],
            "query": text.get(k[0], ""),
            "category": query_category[k[0]],
            "image": k[1],
            "a": x,
            "b": y,
            "abs_diff": abs(x - y),
        }
        for k, x, y in differing
    ]

    by_category: dict[str, dict[str, int]] = {}
    for k in keys:
        c = by_category.setdefault(
            query_category[k[0]],
            {
                "pairs": 0,
                "unsure_excluded": 0,
                "comparable": 0,
                "agree": 0,
                "diff_1": 0,
                "diff_2": 0,
            },
        )
        c["pairs"] += 1
        if k in ua or k in ub:
            c["unsure_excluded"] += 1
            continue
        c["comparable"] += 1
        d = abs(int(la[k].grade) - int(lb[k].grade))  # type: ignore[arg-type]
        c["agree" if d == 0 else f"diff_{d}"] += 1

    return {
        "overlap_pairs": len(keys),
        "unsure": {
            "a": len(ua),
            "b": len(ub),
            "both": len(ua & ub),
            "a_only": len(ua - ub),
            "b_only": len(ub - ua),
            "excluded_total": len(ua | ub),
        },
        "comparable_pairs": len(graded),
        "comparable_queries": len(set(qids)),
        "ordinal": _with_ci(ga, gb, qids, GRADES, n_boot, seed),
        "binary": binary,
        "confusion_rows_a_cols_b": confusion_matrix(ga, gb).tolist(),
        "marginals_comparable": {"a": _distribution(ga), "b": _distribution(gb)},
        "marginals_overlap": {
            "a": _distribution(la[k].grade for k in keys),
            "b": _distribution(lb[k].grade for k in keys),
        },
        "disagreements": disagreements,
        "disagreement_counts": {
            "diff_2": sum(d["abs_diff"] == 2 for d in disagreements),
            "diff_1": sum(d["abs_diff"] == 1 for d in disagreements),
        },
        "by_category": dict(sorted(by_category.items())),
        "bootstrap": {"n": n_boot, "seed": seed, "ci": CI_LEVEL, "units": ["pair", "query"]},
    }


def agreement_limitations(
    detail: Mapping[str, Any], a: str, b: str, extra: Sequence[str] = ()
) -> list[str]:
    """Caveats derived from the numbers in ``detail``, followed by ``extra`` (fixed facts)."""
    n, nq = detail["comparable_pairs"], detail["comparable_queries"]
    k = detail["ordinal"]["kappa"]
    cp, cq = k["ci_pair"], k["ci_query"]
    mc = detail["marginals_comparable"]
    u = detail["unsure"]

    def share(dist: Mapping[str, int]) -> str:
        tot = sum(dist[str(g)] for g in GRADES)
        return " / ".join(f"{g}: {dist[str(g)] / tot:.0%}" if tot else "n/a" for g in GRADES)

    return [
        f"Small sample: {n} comparable pairs from {detail['overlap_pairs']} overlap pairs over "
        f"{nq} queries. Kappa CI width is {cp[1] - cp[0]:.2f} (pair) and "
        f"{cq[1] - cq[0]:.2f} (query); the point estimate alone should not be quoted.",
        "Pairs are not independent: each query contributes several images judged with the "
        "same understanding of that query. The query-clustered CI accounts for this and is "
        "the more honest interval.",
        f"Marginals differ (grade shares on comparable pairs, {a}: {share(mc['a'])}; "
        f"{b}: {share(mc['b'])}). Kappa depends on prevalence and on each annotator's "
        "marginals, so the same raw agreement can give quite different kappa values; read "
        "it together with the raw agreement and the confusion matrix.",
        f"Unsure exclusion is not random: {u['a']} unsure by {a}, {u['b']} by {b} "
        f"({u['both']} by both), {u['excluded_total']} excluded in total. If unsure marks "
        "the harder pairs (plausible, not verified), kappa is biased upwards.",
        "Computed before adjudication on raw labels; it measures reproducibility of the "
        "grading guideline, not correctness of the labels.",
        *extra,
    ]


def _fmt(x: float) -> str:
    return "n/a" if np.isnan(x) else f"{x:.3f}"


def _stat_row(label: str, e: Mapping[str, Any]) -> str:
    cp, cq = e["ci_pair"], e["ci_query"]
    return (
        f"| {label} | {_fmt(e['value'])} | [{_fmt(cp[0])}, {_fmt(cp[1])}] "
        f"| [{_fmt(cq[0])}, {_fmt(cq[1])}] |"
    )


def render_agreement_markdown(doc: Mapping[str, Any]) -> str:
    """Markdown view of the dict built by the agreement CLI (``overlap_agreement`` + meta)."""
    a, b = doc["annotators"]["a"], doc["annotators"]["b"]
    u = doc["unsure"]
    boot = doc["bootstrap"]
    lines = [
        f"# Agreement: {doc['golden']} overlap, {a} vs {b}",
        "",
        f"- command: `{doc['command']}`",
        f"- git: `{doc['git_sha']}` · bootstrap n={boot['n']}, seed={boot['seed']}, "
        f"{boot['ci']:.0%} percentile CI; units: pair (independent pairs) and query "
        "(whole queries resampled)",
        "- labels as recorded, before adjudication; no qrels written",
    ]
    if "adjudication_notes" in doc:
        n = doc["adjudication_notes"]
        line = (
            f"- adjudication: {n['note_source']} notes were optional ({n['policy']}); "
            f"opened for {n['note_shown_pairs']} of {n['adjudicated_pairs']} adjudicated pairs"
        )
        if n["note_seen_unknown_pairs"]:
            line += f" ({n['note_seen_unknown_pairs']} pairs have no note_seen record)"
        lines.append(line)
    lines += [
        "",
        "## 1. Pairs",
        "",
        "| | count |",
        "|---|---|",
        f"| overlap pairs | {doc['overlap_pairs']} |",
        f"| unsure by {a} | {u['a']} |",
        f"| unsure by {b} | {u['b']} |",
        f"| unsure by both | {u['both']} |",
        f"| excluded (unsure on either side) | {u['excluded_total']} |",
        f"| comparable pairs | {doc['comparable_pairs']} ({doc['comparable_queries']} queries) |",
        "",
        "## 2. Ordinal agreement (grades 0/1/2)",
        "",
        "| stat | value | 95% CI (pair) | 95% CI (query) |",
        "|---|---|---|---|",
    ]
    o = doc["ordinal"]
    lines += [
        _stat_row("raw agreement", o["observed_agreement"]),
        _stat_row("Cohen's kappa", o["kappa"]),
        _stat_row("linear weighted kappa", o["kappa_linear"]),
        "",
        "## 3. Binary thresholds",
        "",
        "| threshold | stat | value | 95% CI (pair) | 95% CI (query) |",
        "|---|---|---|---|---|",
    ]
    for name, e in doc["binary"].items():
        tag = f"{name} (grade >= {e['threshold']})"
        lines.append(_stat_row(f"{tag} | raw agreement", e["observed_agreement"]))
        lines.append(_stat_row(f"{tag} | kappa", e["kappa"]))
    lines += ["", "Binary confusion (rows = " + a + ", cols = " + b + "; 0 / 1):", ""]
    for name, e in doc["binary"].items():
        m = e["confusion_rows_a_cols_b"]
        lines.append(f"- {name}: `{m[0]}` / `{m[1]}`")
    m = doc["confusion_rows_a_cols_b"]
    mc, mo = doc["marginals_comparable"], doc["marginals_overlap"]
    lines += [
        "",
        "## 4. Confusion matrix and marginals",
        "",
        f"Comparable pairs, rows = {a}, columns = {b}.",
        "",
        f"| {a} \\ {b} | 0 | 1 | 2 | total |",
        "|---|---|---|---|---|",
    ]
    for g, row in zip(GRADES, m, strict=True):
        lines.append(f"| **{g}** | " + " | ".join(map(str, row)) + f" | {sum(row)} |")
    cols = [sum(r[j] for r in m) for j in range(len(GRADES))]
    lines += [
        "| **total** | " + " | ".join(map(str, cols)) + f" | {sum(cols)} |",
        "",
        "| annotator | set | 0 | 1 | 2 | unsure |",
        "|---|---|---|---|---|---|",
    ]
    for who in ("a", "b"):
        name = doc["annotators"][who]
        for label, dist in (("comparable", mc[who]), ("all overlap", mo[who])):
            lines.append(
                f"| {name} | {label} | {dist['0']} | {dist['1']} | {dist['2']} | {dist['unsure']} |"
            )
    dc = doc["disagreement_counts"]
    lines += [
        "",
        "## 5. Disagreements",
        "",
        f"{dc['diff_2'] + dc['diff_1']} disagreements: |{a}-{b}|=2: {dc['diff_2']}, "
        f"|{a}-{b}|=1: {dc['diff_1']}. Sorted by |diff|, then qid.",
        "",
        "By query category (all overlap pairs):",
        "",
        "| category | pairs | unsure excl. | comparable | agree | diff 1 | diff 2 |",
        "|---|---|---|---|---|---|---|",
    ]
    for cat, c in doc["by_category"].items():
        lines.append(
            f"| {cat} | {c['pairs']} | {c['unsure_excluded']} | {c['comparable']} | "
            f"{c['agree']} | {c['diff_1']} | {c['diff_2']} |"
        )
    lines += [
        "",
        f"| abs diff | qid | query | category | image | {a} | {b} |",
        "|---|---|---|---|---|---|---|",
    ]
    for d in doc["disagreements"]:
        lines.append(
            f"| {d['abs_diff']} | {d['qid']} | {d['query']} | {d['category']} | "
            f"`{d['image']}` | {d['a']} | {d['b']} |"
        )
    lines += ["", "## 6. Limitations", ""] + [f"- {x}" for x in doc["limitations"]] + [""]
    return "\n".join(lines)


def final_qrels(primary: Iterable[Judgment], adjudication: Iterable[Judgment]) -> Qrels:
    """Primary annotator's latest grades, overridden by adjudication; ``unsure`` dropped."""
    merged: Mapping[tuple[str, str], Judgment] = {
        **latest_judgments(primary),
        **latest_judgments(adjudication),
    }
    qrels: Qrels = {}
    for (qid, image), j in sorted(merged.items()):
        if j.grade is not None:
            qrels.setdefault(qid, {})[image] = j.grade
    return qrels
