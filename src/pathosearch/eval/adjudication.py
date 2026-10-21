"""Adjudication queue for overlap disagreements (UI lives in ``adjudicate_app.py``).

Reads the disagreements of an agreement report (``reports/agreement_<golden>.json``) and
the second annotator's one-sentence notes, never writes to either annotator's log. Final
decisions go to ``adjudication.jsonl`` through :class:`LabelStore` (i.e.
:func:`append_judgment`) as ``annotator="adjudicated"``, ``caption_seen=False``.

The report is checked against the current judgment logs: if a label changed after the
report was written, loading fails instead of showing stale labels.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from pathosearch.config import require_file
from pathosearch.eval.golden import (
    GoldenPaths,
    Judgment,
    latest_judgments,
    load_judgments,
    read_jsonl,
)
from pathosearch.eval.labeling import LabelStore

ADJUDICATOR = "adjudicated"


@dataclass(frozen=True)
class DisagreementItem:
    """One disagreeing (query, image) pair with both labels and B's note."""

    qid: str
    query: str
    category: str
    image: str
    a: int
    b: int
    note: str | None

    @property
    def key(self) -> tuple[str, str]:
        return (self.qid, self.image)

    @property
    def abs_diff(self) -> int:
        return abs(self.a - self.b)


def load_notes(path: Path) -> dict[tuple[str, str], str]:
    """``(qid, image) -> note``; a pair noted twice is an error."""
    require_file(path, "Expected <golden>/judgments/<annotator>_notes.jsonl.")
    notes: dict[tuple[str, str], str] = {}
    for row in read_jsonl(path):
        key = (row["qid"], row["image"])
        if key in notes:
            raise ValueError(f"duplicate note for {key}")
        notes[key] = row["note"]
    return notes


def check_report_matches_logs(
    items: list[DisagreementItem],
    a_latest: Mapping[tuple[str, str], Judgment],
    b_latest: Mapping[tuple[str, str], Judgment],
) -> None:
    """Raise if any label in the report differs from the annotator's latest judgment."""
    for it in items:
        for side, grade, latest in (("a", it.a, a_latest), ("b", it.b, b_latest)):
            j = latest.get(it.key)
            if j is None or j.grade != grade:
                found = None if j is None else j.grade
                raise ValueError(
                    f"report is stale for {it.key}: {side}={grade} in report, "
                    f"{found} in log. Re-run scripts/agreement_report.py."
                )


def load_disagreements(report_path: Path, golden: GoldenPaths) -> list[DisagreementItem]:
    """Report disagreements sorted by |a - b| (largest first, report order within ties)."""
    require_file(report_path, "Run scripts/agreement_report.py first.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    a_id, b_id = report["annotators"]["a"], report["annotators"]["b"]
    notes = load_notes(golden.judgments_dir / f"{b_id}_notes.jsonl")
    items = [
        DisagreementItem(
            qid=d["qid"],
            query=d["query"],
            category=d["category"],
            image=d["image"],
            a=d["a"],
            b=d["b"],
            note=notes.get((d["qid"], d["image"])),
        )
        for d in report["disagreements"]
    ]
    check_report_matches_logs(
        items,
        latest_judgments(load_judgments(require_file(golden.judgments(a_id), ""))),
        latest_judgments(load_judgments(require_file(golden.judgments(b_id), ""))),
    )
    return sorted(items, key=lambda it: -it.abs_diff)


def adjudication_store(golden: GoldenPaths) -> LabelStore:
    """Append-only store for final decisions (last entry per pair wins)."""
    return LabelStore(golden.adjudication, ADJUDICATOR)


def decide(store: LabelStore, item: DisagreementItem, grade: int | None) -> Judgment:
    """Record the final grade (``None`` = unsure); captions are never shown here."""
    return store.record(item.key, grade, caption_seen=False)
