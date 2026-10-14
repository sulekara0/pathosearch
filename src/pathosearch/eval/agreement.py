"""Inter-annotator agreement (Cohen's kappa) and adjudication.

Workflow: annotator A judges the whole pool, annotator B the overlap subset. Kappa
is computed on pairs both judged with a grade (``unsure`` on either side is excluded
and counted) *before* adjudication. Disagreements are then resolved into
``adjudication.jsonl``; final qrels = A's latest judgments overridden by the
adjudicated ones.

Weights: ``"none"`` (plain Cohen's kappa) or ``"linear"`` (ordinal 0/1/2: a 0-vs-2
disagreement costs twice a 0-vs-1). Kappa is NaN when undefined (no pairs, or
chance agreement is total, e.g. both annotators used a single grade).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from pathosearch.eval.golden import GRADES, Judgment, Qrels, latest_judgments

Weights = Literal["none", "linear"]


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
