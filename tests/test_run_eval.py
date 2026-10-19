import json
from pathlib import Path

import faiss
import numpy as np
import pytest

from pathosearch.eval.golden import (
    CorpusEntry,
    GoldenPaths,
    GoldenQuery,
    Judgment,
    append_judgment,
    save_corpus,
    save_queries,
    write_trec_qrels,
)
from pathosearch.eval.run_eval import (
    caption_seen_exclusions,
    check_split_allowed,
    label_caption_correlation,
    make_runner,
    parse_system,
    render_markdown,
    run_eval,
    score_runs,
    spearman,
    write_report,
)
from pathosearch.eval.splits import save_splits
from pathosearch.pipeline import SearchIndex

IMAGES = [f"img{i}.jpg" for i in range(8)]
QUERIES = [
    GoldenQuery("q1", "squamous carcinoma", "dermatopathology", "t"),
    GoldenQuery("q2", "gland formation", "gastrointestinal", "t"),
    GoldenQuery("q3", "alveolar spaces", "pulmonary", "t"),
    GoldenQuery("q4", "spindle cells", "soft tissue", "t"),
]


def make_golden(tmp_path: Path) -> GoldenPaths:
    g = GoldenPaths(tmp_path / "golden_t")
    save_queries(g.queries, QUERIES)
    save_corpus(g.corpus, [CorpusEntry(i, "unknown", "val_only") for i in IMAGES])
    save_splits(g.splits, {"dev": ["q1", "q2", "q3"], "test": ["q4"]}, seed=42, dev_frac=0.75)
    qrels = {
        "q1": {"img0.jpg": 2, "img1.jpg": 1, "img2.jpg": 0},
        "q2": {"img3.jpg": 2, "img4.jpg": 0},
        "q3": {"img5.jpg": 0},  # no relevant image -> NaN
        "q4": {"img6.jpg": 2},
    }
    write_trec_qrels(g.qrels, qrels)
    for qid, per_image in qrels.items():
        for image, grade in per_image.items():
            seen = (qid, image) == ("q1", "img0.jpg")
            append_judgment(g.judgments("A"), Judgment(qid, image, grade, "A", seen, "t"))
    append_judgment(g.judgments("A"), Judgment("q2", "img7.jpg", None, "A", False, "t"))
    return g


ORACLE = {
    "q1": ["img0.jpg", "img1.jpg", "img2.jpg"],
    "q2": ["img3.jpg", "img4.jpg", "img0.jpg"],
    "q3": ["img5.jpg", "img0.jpg", "img1.jpg"],
}
WORST = {
    "q1": ["img2.jpg", "img7.jpg", "img0.jpg"],
    "q2": ["img4.jpg", "img7.jpg", "img3.jpg"],
    "q3": ["img5.jpg", "img0.jpg", "img1.jpg"],
}


def runners() -> dict:
    text_to_run = {QUERIES[i].text: ORACLE[f"q{i + 1}"] for i in range(3)}
    bad = {QUERIES[i].text: WORST[f"q{i + 1}"] for i in range(3)}
    return {"clip": lambda q: text_to_run[q], "quilt": lambda q: bad[q]}


def test_parse_system() -> None:
    assert parse_system("quilt+rerank").rerank and parse_system("clip").encoder == "clip"
    for bad in ("foo", "clip+bm25", "clip+"):
        with pytest.raises(ValueError):
            parse_system(bad)


def test_test_split_needs_the_flag() -> None:
    check_split_allowed("dev", False)
    check_split_allowed("test", True)
    with pytest.raises(ValueError, match="i-know-this-is-test"):
        check_split_allowed("test", False)
    with pytest.raises(ValueError):
        check_split_allowed("train", True)


def test_run_eval_end_to_end(tmp_path: Path) -> None:
    g = make_golden(tmp_path)
    captions = {i: "squamous cell" if i == "img0.jpg" else "" for i in IMAGES}
    report = run_eval(
        g,
        "dev",
        ["clip", "quilt"],
        3,
        runners(),
        captions,
        "cmd x",
        n_boot=200,
        sha="abc1234",
        packages={"numpy": "1"},
    )
    assert report["split_stamp"] == "dev" and report["n_queries"] == 3
    assert report["command"] == "cmd x" and report["git_sha"] == "abc1234"
    assert report["unsure_pairs"] == 1
    assert any("unverified" in c and "QuiltNet" in c for c in report["caveats"])

    strict = report["metrics"]["strict"]
    assert strict["n_queries_without_relevant"]["p"] == 1  # q3 has no grade-2 image
    clip, quilt = strict["systems"]["clip"], strict["systems"]["quilt"]
    assert clip["hit"]["mean"] == 1.0 and clip["hit"]["n"] == 2
    assert quilt["hit"]["mean"] == 1.0  # q1: img0 at rank 3, q2: img3 at rank 3 (k=3)
    assert clip["mrr"]["mean"] == 1.0 and quilt["mrr"]["mean"] == pytest.approx(1 / 3, abs=1e-4)
    assert clip["ndcg"]["mean"] > quilt["ndcg"]["mean"]
    diff = quilt["ndcg"]["diff_vs_first"]
    assert diff["mean"] < 0
    assert "diff_vs_first" not in clip["ndcg"]

    lenient = report["metrics"]["lenient"]["systems"]["clip"]
    assert lenient["p"]["n"] == 2  # q1 and q2 have a grade >= 1 image
    assert set(strict["by_category"]) == {"dermatopathology", "gastrointestinal", "pulmonary"}
    # caption-unseen: the (q1, img0) pair (caption opened) is removed, q1 loses its grade-2 image
    unseen = report["metrics_caption_unseen"]
    assert (unseen["pairs_total"], unseen["pairs_removed"]) == (6, 1)
    assert unseen["share_removed"] == pytest.approx(1 / 6, abs=1e-4) and unseen["warning"] is None
    assert unseen["queries_total"] == 3
    assert unseen["queries_with_relevant_remaining"] == {"strict": 1, "lenient": 2}
    assert unseen["strict"]["n_queries_without_relevant"]["p"] == 2
    assert report["circularity"]["n_pairs"] == 6


def test_bootstrap_is_seeded_and_ci_contains_mean(tmp_path: Path) -> None:
    g = make_golden(tmp_path)
    kwargs = dict(n_boot=300, sha="x", packages={})
    a = run_eval(g, "dev", ["clip", "quilt"], 3, runners(), {}, "c", **kwargs)
    b = run_eval(g, "dev", ["clip", "quilt"], 3, runners(), {}, "c", **kwargs)
    assert a == b
    e = a["metrics"]["lenient"]["systems"]["clip"]["ndcg"]
    assert e["ci"][0] <= e["mean"] <= e["ci"][1]


def test_test_split_requires_flag_and_is_stamped(tmp_path: Path) -> None:
    g = make_golden(tmp_path)
    r = {"clip": lambda q: ["img6.jpg", "img0.jpg"]}
    with pytest.raises(ValueError, match="i-know-this-is-test"):
        run_eval(g, "test", ["clip"], 2, r, {}, "c", sha="x", packages={})
    report = run_eval(
        g,
        "test",
        ["clip"],
        2,
        r,
        {},
        "c",
        i_know_this_is_test=True,
        n_boot=50,
        sha="x",
        packages={},
    )
    assert report["split_stamp"] == "TEST" and report["n_queries"] == 1
    assert "TEST SPLIT" in render_markdown(report)


def test_missing_runner_and_duplicates(tmp_path: Path) -> None:
    g = make_golden(tmp_path)
    with pytest.raises(ValueError, match="no runner"):
        run_eval(g, "dev", ["clip", "quilt+rerank"], 3, runners(), {}, "c", sha="x", packages={})
    with pytest.raises(ValueError, match="duplicates"):
        run_eval(g, "dev", ["clip", "clip"], 3, runners(), {}, "c", sha="x", packages={})


def test_write_report_names_and_content(tmp_path: Path) -> None:
    g = make_golden(tmp_path)
    report = run_eval(
        g, "dev", ["clip"], 3, runners(), {}, "cmd", n_boot=50, sha="abc", packages={}
    )
    j, m = write_report(report, tmp_path / "reports")
    assert j.name == "eval_golden_t_dev_abc.json" and m.suffix == ".md"
    loaded = json.loads(j.read_text(encoding="utf-8"))
    for key in ("command", "config", "git_sha", "packages", "caveats", "circularity"):
        assert key in loaded
    assert "cmd" in m.read_text(encoding="utf-8")


def test_caption_seen_exclusions() -> None:
    qrels = {"q": {"a": 2, "b": 1, "c": 0}}
    js = [
        Judgment("q", "a", 2, "A", True, "t"),
        Judgment("q", "b", 1, "A", False, "t"),
    ]  # c has no reference judgment -> exposure unknown -> excluded
    kept, excluded = caption_seen_exclusions(qrels, js)
    assert kept == {"q": {"b": 1}} and excluded == {"q": {"a", "c"}}


def test_excluded_images_leave_the_ranked_list_instead_of_counting_as_misses() -> None:
    runs = {"s": {"q": ["a", "b"]}}
    qrels = {"q": {"b": 2}}
    plain = score_runs(runs, qrels, 2)["strict"]["s"]
    condensed = score_runs(runs, qrels, 2, excluded={"q": {"a"}})["strict"]["s"]
    assert plain["p"]["q"] == 0.5 and plain["mrr"]["q"] == 0.5
    assert condensed["p"]["q"] == 1.0 and condensed["mrr"]["q"] == 1.0  # list is [b], scored at 1
    emptied = score_runs(runs, qrels, 2, excluded={"q": {"a", "b"}})["strict"]["s"]
    assert all(v["q"] != v["q"] for v in emptied.values())  # NaN: nothing left to score


def test_warns_when_too_many_pairs_are_removed(tmp_path: Path) -> None:
    g = make_golden(tmp_path)
    for qid, image in [
        ("q1", "img0.jpg"),
        ("q1", "img1.jpg"),
        ("q2", "img3.jpg"),
        ("q2", "img4.jpg"),
    ]:
        append_judgment(g.judgments("A"), Judgment(qid, image, 1, "A", True, "t"))  # caption opened
    report = run_eval(g, "dev", ["clip"], 3, runners(), {}, "c", n_boot=50, sha="x", packages={})
    cu = report["metrics_caption_unseen"]
    assert cu["pairs_removed"] == 4 and cu["share_removed"] > 0.30 and "removed" in cu["warning"]
    assert "**Warning:**" in render_markdown(report)


def test_spearman() -> None:
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2, 2, 3], [1, 2, 2, 3]) == pytest.approx(1.0)  # ties
    assert spearman([1, 1, 1], [1, 2, 3]) is None  # constant
    assert spearman([1, 2], [1, 2]) is None  # too few
    with pytest.raises(ValueError):
        spearman([1, 2, 3], [1, 2])


def test_label_caption_correlation_warns_when_labels_follow_wording() -> None:
    queries = {q.qid: q for q in QUERIES}
    qrels = {"q1": {"a": 2, "b": 1, "c": 0, "d": 0}}
    dependent = {
        "a": "squamous carcinoma",
        "b": "squamous",
        "c": "unrelated words",
        "d": "other text",
    }
    res = label_caption_correlation(qrels, queries, dependent)
    assert res["spearman_rho"] > 0.5 and res["warning"]
    flat = label_caption_correlation(qrels, queries, {k: "same" for k in "abcd"})
    assert flat["spearman_rho"] is None and flat["warning"] is None


class VecEncoder:
    def __init__(self, vec: list[float]) -> None:
        self._v = np.array([vec], dtype=np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return self._v


def test_make_runner_uses_index_and_clamps_pool() -> None:
    vecs = np.eye(4, dtype=np.float32)
    index = faiss.IndexFlatIP(4)
    index.add(vecs)
    ids = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    caps = {
        "a.jpg": {"caption": "", "pathology": "x"},
        "b.jpg": {"caption": "squamous carcinoma", "pathology": "x"},
    }
    idx = SearchIndex(index, "ip", ids, caps)
    enc = VecEncoder([1.0, 0.98, 0.0, 0.0])  # a and b almost tied by cosine
    assert make_runner(enc, idx, False, 2)("squamous carcinoma")[0] == "a.jpg"
    reranked = make_runner(enc, idx, True, 2)("squamous carcinoma")  # pool 10 > ntotal 4
    assert len(reranked) == 2 and reranked[0] == "b.jpg"  # caption overlap lifts b above a
