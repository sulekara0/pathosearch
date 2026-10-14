from pathlib import Path

import pytest

from pathosearch.errors import DataNotFoundError
from pathosearch.eval.golden import (
    CorpusEntry,
    GoldenPaths,
    GoldenQuery,
    Judgment,
    append_judgment,
    judgments_to_qrels,
    latest_judgments,
    load_corpus,
    load_judgments,
    load_queries,
    read_trec_qrels,
    save_corpus,
    save_queries,
    write_trec_qrels,
)


def j(qid: str, image: str, grade: int | None, annotator: str = "A", ts: str = "t") -> Judgment:
    return Judgment(qid, image, grade, annotator, caption_seen=False, ts=ts)


def test_queries_round_trip_and_category_normalized(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    qs = [
        GoldenQuery("q1", "melanoma", "Dermatopathology ", "hand-written"),
        GoldenQuery("q2", "colon adenocarcinoma", "gastrointestinal", "hand-written"),
    ]
    save_queries(path, qs)
    loaded = load_queries(path)
    assert loaded == qs
    assert loaded[0].category == "dermatopathology"


def test_duplicate_qid_rejected(tmp_path: Path) -> None:
    qs = [GoldenQuery("q1", "a", "x", "p"), GoldenQuery("q1", "b", "x", "p")]
    with pytest.raises(ValueError, match="duplicate qid"):
        save_queries(tmp_path / "q.jsonl", qs)


def test_empty_query_text_rejected() -> None:
    with pytest.raises(ValueError):
        GoldenQuery("q1", "  ", "x", "p")


def test_corpus_round_trip_and_duplicate_image_rejected(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    entries = [
        CorpusEntry("a.jpg", "Unknown", "val_only"),
        CorpusEntry("b.jpg", "renal", "val_only"),
    ]
    save_corpus(path, entries)
    loaded = load_corpus(path)
    assert loaded == entries
    assert loaded[0].pathology == "unknown"
    with pytest.raises(ValueError, match="duplicate image"):
        save_corpus(path, [entries[0], entries[0]])


@pytest.mark.parametrize("grade", [-1, 3, 5])
def test_invalid_grade_rejected(grade: int) -> None:
    with pytest.raises(ValueError, match="grade"):
        j("q1", "a.jpg", grade)


def test_append_only_log_last_judgment_wins(tmp_path: Path) -> None:
    path = GoldenPaths(tmp_path).judgments("A")
    append_judgment(path, j("q1", "a.jpg", 0, ts="1"))
    append_judgment(path, j("q1", "b.jpg", 2, ts="2"))
    append_judgment(path, j("q1", "a.jpg", 1, ts="3"))  # correction
    log = load_judgments(path)
    assert len(log) == 3  # nothing overwritten
    latest = latest_judgments(log)
    assert latest[("q1", "a.jpg")].grade == 1
    assert latest[("q1", "a.jpg")].ts == "3"


def test_unsure_is_dropped_from_qrels() -> None:
    qrels = judgments_to_qrels([j("q1", "a.jpg", 2), j("q1", "b.jpg", None), j("q2", "a.jpg", 0)])
    assert qrels == {"q1": {"a.jpg": 2}, "q2": {"a.jpg": 0}}


def test_unsure_overrides_earlier_grade() -> None:
    assert judgments_to_qrels([j("q1", "a.jpg", 2), j("q1", "a.jpg", None)]) == {}


def test_trec_qrels_round_trip_sorted(tmp_path: Path) -> None:
    path = tmp_path / "qrels.tsv"
    qrels = {"q2": {"b.jpg": 1}, "q1": {"z.jpg": 0, "a.jpg": 2}}
    write_trec_qrels(path, qrels)
    assert path.read_text(encoding="utf-8").splitlines() == [
        "q1 0 a.jpg 2",
        "q1 0 z.jpg 0",
        "q2 0 b.jpg 1",
    ]
    assert read_trec_qrels(path) == qrels


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("q1 0 a.jpg\n", "expected"),
        ("q1 0 a.jpg 3\n", "grade"),
        ("q1 0 a.jpg 1\nq1 0 a.jpg 2\n", "duplicate"),
    ],
)
def test_malformed_trec_qrels_rejected(tmp_path: Path, content: str, match: str) -> None:
    path = tmp_path / "qrels.tsv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        read_trec_qrels(path)


def test_missing_files_raise_data_not_found(tmp_path: Path) -> None:
    paths = GoldenPaths(tmp_path / "golden_v0")
    with pytest.raises(DataNotFoundError):
        load_queries(paths.queries)
    with pytest.raises(DataNotFoundError):
        read_trec_qrels(paths.qrels)
