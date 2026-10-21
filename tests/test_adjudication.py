import json
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from pathosearch.errors import DataNotFoundError
from pathosearch.eval.adjudication import (
    ADJUDICATOR,
    adjudication_store,
    decide,
    load_disagreements,
    load_notes,
)
from pathosearch.eval.fake_golden import make_fake_golden
from pathosearch.eval.golden import GoldenPaths, Judgment, append_judgment, read_jsonl
from pathosearch.eval.pooling import load_pool

APP = Path(__file__).resolve().parent.parent / "adjudicate_app.py"


@pytest.fixture
def setup(tmp_path: Path):
    """Fake golden + A/claude logs + notes + a report with three disagreements."""
    fake = make_fake_golden(tmp_path, n_images=30, n_queries=3, depth=3, n_random=1)
    golden = GoldenPaths(fake.golden_dir)
    pairs = [(it.qid, it.image) for it in load_pool(golden.pool)][:4]
    a_grades = [1, 0, 2, 1]
    b_grades = [0, 2, 1, 1]  # diffs 1, 2, 1, 0
    for (qid, img), ga, gb in zip(pairs, a_grades, b_grades, strict=True):
        append_judgment(golden.judgments("A"), Judgment(qid, img, ga, "A", False, "t"))
        append_judgment(golden.judgments("claude"), Judgment(qid, img, gb, "claude", False, "t"))
    notes = golden.judgments_dir / "claude_notes.jsonl"
    with notes.open("w", encoding="utf-8") as fh:
        for qid, img in pairs[:2]:
            fh.write(json.dumps({"qid": qid, "image": img, "note": f"note {img}"}) + "\n")
    disagreements = [
        {"qid": q, "query": f"text {q}", "category": "c", "image": i, "a": ga, "b": gb}
        for (q, i), ga, gb in zip(pairs, a_grades, b_grades, strict=True)
        if ga != gb
    ]
    report = tmp_path / "agreement.json"
    report.write_text(
        json.dumps({"annotators": {"a": "A", "b": "claude"}, "disagreements": disagreements}),
        encoding="utf-8",
    )
    return fake, golden, report, pairs


def test_sorted_by_diff_with_notes(setup) -> None:
    _, golden, report, pairs = setup
    items = load_disagreements(report, golden)
    assert [it.key for it in items] == [pairs[1], pairs[0], pairs[2]]  # stable within ties
    assert [it.abs_diff for it in items] == [2, 1, 1]
    assert items[0].note == f"note {pairs[1][1]}"
    assert items[2].note is None  # no note for that pair: shown as missing, not invented


def test_stale_report_is_rejected(setup) -> None:
    _, golden, report, pairs = setup
    qid, img = pairs[0]
    append_judgment(golden.judgments("A"), Judgment(qid, img, 0, "A", False, "t2"))
    with pytest.raises(ValueError, match="stale"):
        load_disagreements(report, golden)


def test_missing_notes_file_is_an_error(setup) -> None:
    _, golden, report, _ = setup
    (golden.judgments_dir / "claude_notes.jsonl").unlink()
    with pytest.raises(DataNotFoundError):
        load_disagreements(report, golden)


def test_duplicate_note_is_an_error(tmp_path: Path) -> None:
    p = tmp_path / "n.jsonl"
    row = json.dumps({"qid": "q", "image": "x", "note": "n"})
    p.write_text(row + "\n" + row + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_notes(p)


def test_decide_appends_to_adjudication_only(setup) -> None:
    _, golden, report, _ = setup
    before = {p: p.read_bytes() for p in golden.judgments_dir.iterdir()}
    items = load_disagreements(report, golden)
    store = adjudication_store(golden)
    decide(store, items[0], 2)
    decide(store, items[0], None)  # re-decision appends; last wins
    rows = list(read_jsonl(golden.adjudication))
    assert [(r["grade"], r["annotator"], r["caption_seen"]) for r in rows] == [
        (2, ADJUDICATOR, False),
        (None, ADJUDICATOR, False),
    ]
    assert {p: p.read_bytes() for p in golden.judgments_dir.iterdir()} == before


def test_app_decides_and_advances(setup, monkeypatch: pytest.MonkeyPatch) -> None:
    fake, golden, report, pairs = setup
    argv = ["adjudicate_app.py", "--golden", str(fake.golden_dir), "--report", str(report)]
    argv += ["--images-dir", str(fake.images_dir)]
    monkeypatch.setattr(sys, "argv", argv)
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert any("1/3" in m.value for m in at.markdown)
    at.button(key="decide_1").click().run()
    assert not at.exception and any("2/3" in m.value for m in at.markdown)
    at.button(key="decide_unsure").click().run()
    rows = list(read_jsonl(golden.adjudication))
    assert [(r["qid"], r["image"], r["grade"]) for r in rows] == [
        (*pairs[1], 1),
        (*pairs[0], None),
    ]


def test_app_reports_missing_report(setup, monkeypatch: pytest.MonkeyPatch) -> None:
    fake, _, report, _ = setup
    report.unlink()
    argv = ["adjudicate_app.py", "--golden", str(fake.golden_dir), "--report", str(report)]
    monkeypatch.setattr(sys, "argv", argv + ["--images-dir", str(fake.images_dir)])
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert any("okunamadı" in e.value for e in at.error)
