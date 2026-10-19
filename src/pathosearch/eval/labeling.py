"""State and persistence for the golden-set labeling tool (UI lives in ``label_app.py``).

Judgments go to ``judgments/<annotator>.jsonl`` through :func:`append_judgment`: the log is
append-only and the last entry per (qid, image) wins. "Undo" therefore never edits the log;
it steps back to the previously judged pair so it can be judged again (the new row wins).

Annotators never see which system retrieved an image or in what rank, only the pool's
shuffled ``order``. The primary annotator judges the whole pool; every other annotator
judges only the overlap subset.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pathosearch.config import require_dir, require_file
from pathosearch.errors import DataNotFoundError
from pathosearch.eval.golden import (
    GoldenPaths,
    GoldenQuery,
    Judgment,
    append_judgment,
    latest_judgments,
    load_judgments,
    load_queries,
)
from pathosearch.eval.pooling import PoolItem, load_pool

PRIMARY_ANNOTATOR = "A"


@dataclass(frozen=True)
class Task:
    """One (query, image) pair to judge."""

    query: GoldenQuery
    item: PoolItem

    @property
    def key(self) -> tuple[str, str]:
        return (self.item.qid, self.item.image)


def select_items(
    pool: Sequence[PoolItem], overlap: Sequence[PoolItem], annotator: str
) -> list[PoolItem]:
    """Whole pool for the primary annotator, overlap subset for everyone else."""
    return list(pool) if annotator == PRIMARY_ANNOTATOR else list(overlap)


def build_queue(queries: Iterable[GoldenQuery], items: Iterable[PoolItem]) -> list[Task]:
    """Tasks grouped by query (in ``queries.jsonl`` order), shuffled ``order`` within a query."""
    by_qid = {q.qid: q for q in queries}
    grouped: dict[str, list[PoolItem]] = {}
    for it in items:
        if it.qid not in by_qid:
            raise ValueError(f"pool item refers to unknown qid {it.qid!r}")
        grouped.setdefault(it.qid, []).append(it)
    return [
        Task(by_qid[qid], it)
        for qid in by_qid
        for it in sorted(grouped.get(qid, []), key=lambda x: x.order)
    ]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class LabelStore:
    """Judgment log of one annotator (append-only, last entry per pair wins)."""

    def __init__(self, path: Path, annotator: str) -> None:
        self.path = path
        self.annotator = annotator
        self._latest: dict[tuple[str, str], Judgment] = (
            latest_judgments(load_judgments(path)) if path.is_file() else {}
        )

    def get(self, key: tuple[str, str]) -> Judgment | None:
        return self._latest.get(key)

    def record(
        self,
        key: tuple[str, str],
        grade: int | None,
        caption_seen: bool,
        ts: str | None = None,
    ) -> Judgment:
        j = Judgment(key[0], key[1], grade, self.annotator, caption_seen, ts or now_iso())
        append_judgment(self.path, j)
        self._latest[key] = j
        return j

    def is_done(self, key: tuple[str, str]) -> bool:
        return key in self._latest

    def done_count(self, queue: Sequence[Task]) -> int:
        return sum(1 for t in queue if self.is_done(t.key))


class LabelingSession:
    """Cursor over the queue: next unjudged pair, with a one-step-at-a-time undo."""

    def __init__(
        self,
        queue: Sequence[Task],
        store: LabelStore,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.queue = list(queue)
        self.store = store
        self._clock = clock
        self.started = clock()
        self._history: list[int] = []  # queue indices judged this session, oldest first
        self._revisit: int | None = None
        self.n_submitted = 0

    @property
    def current_index(self) -> int | None:
        if self._revisit is not None:
            return self._revisit
        for i, task in enumerate(self.queue):
            if not self.store.is_done(task.key):
                return i
        return None

    @property
    def current(self) -> Task | None:
        i = self.current_index
        return None if i is None else self.queue[i]

    @property
    def revisiting(self) -> bool:
        return self._revisit is not None

    @property
    def done(self) -> int:
        return self.store.done_count(self.queue)

    @property
    def total(self) -> int:
        return len(self.queue)

    @property
    def can_undo(self) -> bool:
        return bool(self._history)

    def submit(self, grade: int | None, caption_seen: bool) -> Judgment:
        """Record the current pair (``grade=None`` = unsure) and advance."""
        i = self.current_index
        if i is None:
            raise RuntimeError("nothing left to judge")
        task = self.queue[i]
        previous = self.store.get(task.key)
        seen = caption_seen or (previous is not None and previous.caption_seen)
        j = self.store.record(task.key, grade, seen)
        self._history.append(i)
        self._revisit = None
        self.n_submitted += 1
        return j

    def undo(self) -> bool:
        """Step back to the previously judged pair; its old label stays until re-judged."""
        if not self._history:
            return False
        self._revisit = self._history.pop()
        return True

    def elapsed(self) -> float:
        return self._clock() - self.started

    def seconds_per_label(self) -> float | None:
        return self.elapsed() / self.n_submitted if self.n_submitted else None


def format_elapsed(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def load_queue(golden: GoldenPaths, annotator: str) -> list[Task]:
    """Queue for ``annotator`` from a golden dir. Missing files -> DataNotFoundError."""
    queries = load_queries(golden.queries)
    pool = load_pool(golden.pool)
    overlap = load_pool(golden.overlap) if annotator != PRIMARY_ANNOTATOR else []
    return build_queue(queries, select_items(pool, overlap, annotator))


# --- images and captions -------------------------------------------------------


def build_image_index(images_dir: Path) -> dict[str, Path]:
    """File name -> path for every file under ``images_dir`` (recursive).

    Empty/missing directory or two different files with the same name raise: a label must
    never be attached to an ambiguous or wrong image.
    """
    require_dir(images_dir, "Set PATHOSEARCH_IMAGES_DIR in .env or pass --images-dir.")
    index: dict[str, Path] = {}
    for root, dirs, files in os.walk(images_dir):
        dirs.sort()
        for name in sorted(files):
            if name in index:
                raise ValueError(f"image name {name!r} is not unique: {index[name]} vs {root}")
            index[name] = Path(root) / name
    return index


def resolve_image(index: Mapping[str, Path], name: str) -> Path:
    """Path of image ``name``; raises (never substitutes another image) if absent."""
    try:
        return index[name]
    except KeyError:
        raise DataNotFoundError(f"Image not found under the images directory: {name}") from None


def load_first_captions(csv_path: Path, images: Iterable[str]) -> dict[str, str]:
    """First CSV caption per requested image (deterministic; see R2 plan, risks)."""
    require_file(csv_path, "Set PATHOSEARCH_CSV_PATH in .env or pass --csv.")
    wanted = set(images)
    found: dict[str, str] = {}
    for chunk in pd.read_csv(csv_path, usecols=["image_path", "caption"], chunksize=200_000):
        hit = chunk[chunk["image_path"].isin(wanted)]
        for name, caption in zip(hit["image_path"], hit["caption"], strict=True):
            found.setdefault(str(name), "" if pd.isna(caption) else str(caption))
    return found
