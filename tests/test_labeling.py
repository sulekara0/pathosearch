import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from pathosearch.errors import DataNotFoundError
from pathosearch.eval.fake_golden import make_fake_golden
from pathosearch.eval.golden import GoldenPaths, load_judgments, read_jsonl
from pathosearch.eval.labeling import (
    PRIMARY_ANNOTATOR,
    LabelingSession,
    LabelStore,
    build_image_index,
    build_queue,
    format_elapsed,
    load_first_captions,
    load_queue,
    resolve_image,
)
from pathosearch.eval.pooling import PoolItem

APP = Path(__file__).resolve().parent.parent / "label_app.py"


@pytest.fixture
def fake(tmp_path: Path):
    return make_fake_golden(tmp_path, n_images=30, n_queries=3, depth=3, n_random=1)


def make_session(fake, annotator: str = "A", clock=None) -> LabelingSession:
    golden = GoldenPaths(fake.golden_dir)
    store = LabelStore(golden.judgments(annotator), annotator)
    queue = load_queue(golden, annotator)
    return LabelingSession(queue, store, **({"clock": clock} if clock else {}))


def test_queue_groups_by_query_in_pool_order(fake) -> None:
    queue = load_queue(GoldenPaths(fake.golden_dir), "A")
    qids = [t.item.qid for t in queue]
    assert qids == sorted(qids)  # queries.jsonl order, grouped
    for qid in set(qids):
        orders = [t.item.order for t in queue if t.item.qid == qid]
        assert orders == sorted(orders)


def test_non_primary_annotator_sees_only_overlap(fake) -> None:
    full = load_queue(GoldenPaths(fake.golden_dir), PRIMARY_ANNOTATOR)
    sub = load_queue(GoldenPaths(fake.golden_dir), "B")
    assert 0 < len(sub) < len(full)
    assert {t.key for t in sub} <= {t.key for t in full}


def test_unknown_qid_in_pool_is_an_error(fake) -> None:
    with pytest.raises(ValueError, match="unknown qid"):
        build_queue([], [PoolItem("nope", "x.jpg", 0, "pooled", ())])


def test_submit_appends_and_resumes(fake) -> None:
    s = make_session(fake)
    first = s.current
    assert first is not None
    s.submit(2, caption_seen=False)
    s.submit(None, caption_seen=True)  # unsure is a recorded judgment
    assert s.done == 2

    rows = list(read_jsonl(GoldenPaths(fake.golden_dir).judgments("A")))
    assert [r["grade"] for r in rows] == [2, None]
    assert rows[0]["annotator"] == "A" and rows[1]["caption_seen"] is True

    resumed = make_session(fake)  # new process: continues after the judged pairs
    assert resumed.done == 2
    assert resumed.current is not None and resumed.current.key == s.current.key


def test_undo_relabels_without_touching_the_log(fake) -> None:
    s = make_session(fake)
    first = s.current
    s.submit(1, caption_seen=True)
    assert s.can_undo and s.undo()
    assert s.revisiting and s.current == first
    s.submit(0, caption_seen=False)  # re-judge: caption_seen stays true once seen
    assert not s.revisiting and s.current != first

    judgments = load_judgments(GoldenPaths(fake.golden_dir).judgments("A"))
    assert [j.grade for j in judgments] == [1, 0]  # append-only
    assert judgments[1].caption_seen is True
    assert s.done == 1  # still one distinct pair


def test_undo_with_nothing_to_undo(fake) -> None:
    s = make_session(fake)
    assert not s.can_undo and not s.undo()


def test_finishing_the_queue(fake) -> None:
    s = make_session(fake, "B")
    for _ in range(s.total):
        s.submit(1, caption_seen=False)
    assert s.current is None and s.done == s.total
    with pytest.raises(RuntimeError):
        s.submit(1, caption_seen=False)


def test_elapsed_and_seconds_per_label(fake) -> None:
    t = [100.0]
    s = make_session(fake, clock=lambda: t[0])
    assert s.seconds_per_label() is None
    t[0] = 130.0
    s.submit(1, caption_seen=False)
    s.submit(1, caption_seen=False)
    assert s.elapsed() == 30.0 and s.seconds_per_label() == 15.0


def test_format_elapsed() -> None:
    assert format_elapsed(5) == "00:05"
    assert format_elapsed(125) == "02:05"
    assert format_elapsed(3725) == "1:02:05"


def test_image_lookup_is_recursive_and_explicit(fake) -> None:
    index = build_image_index(fake.images_dir)
    assert resolve_image(index, "fake_0000.jpg").is_file()
    with pytest.raises(DataNotFoundError):
        resolve_image(index, "missing.jpg")


def test_duplicate_image_names_are_an_error(tmp_path: Path) -> None:
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "x.jpg").write_bytes(b"")
    with pytest.raises(ValueError, match="not unique"):
        build_image_index(tmp_path)


def test_missing_images_dir_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(DataNotFoundError):
        build_image_index(tmp_path / "nope")


def test_captions_use_first_csv_row(fake) -> None:
    caps = load_first_captions(fake.csv_path, ["fake_0000.jpg", "fake_0001.jpg"])
    assert caps["fake_0000.jpg"] == "fake first caption of fake_0000.jpg"  # has a 2nd caption
    assert set(caps) == {"fake_0000.jpg", "fake_0001.jpg"}


# --- AppTest smoke -----------------------------------------------------------------


def run_app(fake, monkeypatch: pytest.MonkeyPatch, annotator: str = "A") -> AppTest:
    argv = ["label_app.py", "--golden", str(fake.golden_dir), "--annotator", annotator]
    argv += ["--images-dir", str(fake.images_dir), "--csv", str(fake.csv_path)]
    monkeypatch.setattr(sys, "argv", argv)
    return AppTest.from_file(str(APP), default_timeout=60).run()


def test_app_labels_a_pair_and_undoes(fake, monkeypatch: pytest.MonkeyPatch) -> None:
    at = run_app(fake, monkeypatch)
    assert not at.exception
    assert any("Sorgu:" in m.value for m in at.markdown)
    shortcuts = {b.key: b.proto.shortcut for b in at.button}
    assert shortcuts["grade_0"] == "0" and shortcuts["grade_2"] == "2"
    assert shortcuts["undo"] == "u"
    assert at.button(key="undo").disabled

    at.button(key="show_caption").click().run()
    assert any("fake first caption of" in i.value for i in at.info)
    at.button(key="grade_2").click().run()
    at.button(key="grade_unsure").click().run()
    assert not at.exception

    rows = list(read_jsonl(GoldenPaths(fake.golden_dir).judgments("A")))
    assert [(r["grade"], r["caption_seen"]) for r in rows] == [(2, True), (None, False)]

    at.button(key="undo").click().run()
    assert any("Geri alınan çift" in i.value for i in at.info)


def test_app_stops_with_clear_error_when_image_missing(
    fake, monkeypatch: pytest.MonkeyPatch
) -> None:
    for p in fake.images_dir.rglob("*.jpg"):
        p.unlink()
    at = run_app(fake, monkeypatch)
    assert not at.exception
    assert len(at.error) == 1 and "Image not found" in at.error[0].value
    assert not any(b.key.startswith("grade_") for b in at.button)  # no labeling without image


def test_app_reports_missing_golden_dir(
    fake, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake2 = type(fake)(tmp_path / "nowhere", fake.images_dir, fake.csv_path)
    at = run_app(fake2, monkeypatch)
    assert not at.exception
    assert any("Golden dizini okunamadı" in e.value for e in at.error)
