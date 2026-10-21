"""Golden set schema and file IO.

A golden set is a versioned directory (e.g. ``eval_data/golden_v0/``). Nothing here
knows how many images or queries it holds; the corpus is whatever the manifest lists.

Layout (see :class:`GoldenPaths`):
    queries.jsonl          GoldenQuery per line
    splits.json            {"dev": [qid, ...], "test": [qid, ...], ...}
    corpus_manifest.jsonl  CorpusEntry per line (image names + metadata, no captions)
    pool.jsonl             pooled (qid, image) pairs to judge
    overlap.jsonl          subset of the pool judged by a second annotator
    judgments/<id>.jsonl   append-only Judgment log per annotator
    adjudication.jsonl     resolved disagreements (Judgment, annotator="adjudicated")
    adjudication_meta.jsonl  per-decision side log (e.g. whether B's note was shown)
    qrels.tsv              final graded qrels, TREC format ``qid 0 image grade``

Grades: 0 = not relevant, 1 = partially relevant, 2 = relevant; ``None`` = unsure
(kept in the log, never written to qrels).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pathosearch.config import require_file
from pathosearch.labels import normalize_pathology

GRADES = (0, 1, 2)

Qrels = dict[str, dict[str, int]]  # qid -> image -> grade


@dataclass(frozen=True)
class GoldenQuery:
    qid: str
    text: str
    category: str
    provenance: str

    def __post_init__(self) -> None:
        if not self.qid or not self.text.strip():
            raise ValueError(f"query needs qid and text: {self!r}")
        object.__setattr__(self, "category", normalize_pathology(self.category))


@dataclass(frozen=True)
class CorpusEntry:
    image: str
    pathology: str
    split_class: str

    def __post_init__(self) -> None:
        if not self.image:
            raise ValueError("corpus entry needs an image name")
        object.__setattr__(self, "pathology", normalize_pathology(self.pathology))


@dataclass(frozen=True)
class Judgment:
    qid: str
    image: str
    grade: int | None
    annotator: str
    caption_seen: bool
    ts: str

    def __post_init__(self) -> None:
        if self.grade is not None and self.grade not in GRADES:
            raise ValueError(f"grade must be one of {GRADES} or None, got {self.grade!r}")
        if not self.annotator:
            raise ValueError("judgment needs an annotator id")

    @property
    def key(self) -> tuple[str, str]:
        return (self.qid, self.image)


@dataclass(frozen=True)
class GoldenPaths:
    root: Path

    @property
    def queries(self) -> Path:
        return self.root / "queries.jsonl"

    @property
    def splits(self) -> Path:
        return self.root / "splits.json"

    @property
    def corpus(self) -> Path:
        return self.root / "corpus_manifest.jsonl"

    @property
    def pool(self) -> Path:
        return self.root / "pool.jsonl"

    @property
    def overlap(self) -> Path:
        return self.root / "overlap.jsonl"

    @property
    def judgments_dir(self) -> Path:
        return self.root / "judgments"

    def judgments(self, annotator: str) -> Path:
        return self.judgments_dir / f"{annotator}.jsonl"

    @property
    def adjudication(self) -> Path:
        return self.root / "adjudication.jsonl"

    @property
    def adjudication_meta(self) -> Path:
        return self.root / "adjudication_meta.jsonl"

    @property
    def qrels(self) -> Path:
        return self.root / "qrels.tsv"


# --- generic JSONL -----------------------------------------------------------


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield one dict per non-empty line. Missing file -> DataNotFoundError."""
    require_file(path)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _unique(keys: list[str], what: str) -> None:
    dupes = sorted(k for k, n in Counter(keys).items() if n > 1)
    if dupes:
        raise ValueError(f"duplicate {what}: {dupes[:5]}")


# --- queries / corpus --------------------------------------------------------


def load_queries(path: Path) -> list[GoldenQuery]:
    queries = [GoldenQuery(**row) for row in read_jsonl(path)]
    _unique([q.qid for q in queries], "qid")
    return queries


def save_queries(path: Path, queries: Iterable[GoldenQuery]) -> None:
    queries = list(queries)
    _unique([q.qid for q in queries], "qid")
    write_jsonl(path, (asdict(q) for q in queries))


def load_corpus(path: Path) -> list[CorpusEntry]:
    entries = [CorpusEntry(**row) for row in read_jsonl(path)]
    _unique([e.image for e in entries], "image")
    return entries


def save_corpus(path: Path, entries: Iterable[CorpusEntry]) -> None:
    entries = list(entries)
    _unique([e.image for e in entries], "image")
    write_jsonl(path, (asdict(e) for e in entries))


# --- judgments ---------------------------------------------------------------


def append_judgment(path: Path, judgment: Judgment) -> None:
    """Append one judgment; the log is never rewritten (last entry per pair wins)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(asdict(judgment), ensure_ascii=False) + "\n")


def load_judgments(path: Path) -> list[Judgment]:
    return [Judgment(**row) for row in read_jsonl(path)]


def latest_judgments(judgments: Iterable[Judgment]) -> dict[tuple[str, str], Judgment]:
    """Last judgment per (qid, image), in log order."""
    latest: dict[tuple[str, str], Judgment] = {}
    for j in judgments:
        latest[j.key] = j
    return latest


def judgments_to_qrels(judgments: Iterable[Judgment]) -> Qrels:
    """Graded qrels from judgments (last wins per pair); ``unsure`` pairs are dropped."""
    qrels: Qrels = {}
    for j in latest_judgments(judgments).values():
        if j.grade is not None:
            qrels.setdefault(j.qid, {})[j.image] = j.grade
    return qrels


# --- TREC qrels --------------------------------------------------------------


def write_trec_qrels(path: Path, qrels: Qrels) -> None:
    """Write ``qid 0 image grade`` lines, sorted for stable diffs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{qid} 0 {image} {grade}\n"
        for qid in sorted(qrels)
        for image, grade in sorted(qrels[qid].items())
    ]
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(lines)


def read_trec_qrels(path: Path) -> Qrels:
    require_file(path)
    qrels: Qrels = {}
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"{path}:{n}: expected 'qid 0 image grade', got {line!r}")
            qid, _, image, grade_s = parts
            grade = int(grade_s)
            if grade not in GRADES:
                raise ValueError(f"{path}:{n}: grade {grade} not in {GRADES}")
            if image in qrels.setdefault(qid, {}):
                raise ValueError(f"{path}:{n}: duplicate pair ({qid}, {image})")
            qrels[qid][image] = grade
    return qrels
