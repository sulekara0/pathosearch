import math

import pytest

from pathosearch.eval.agreement import (
    agreement_limitations,
    agreement_report,
    binarize,
    bootstrap_ci,
    cohen_kappa,
    confusion_matrix,
    final_qrels,
    observed_agreement,
    overlap_agreement,
    render_agreement_markdown,
    unresolved,
)
from pathosearch.eval.golden import Judgment


def j(qid: str, image: str, grade: int | None, annotator: str = "A") -> Judgment:
    return Judgment(qid, image, grade, annotator, caption_seen=False, ts="t")


def test_textbook_binary_kappa() -> None:
    # Classic 2x2 example: 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no -> po=0.7, pe=0.5, k=0.4
    a = [1] * 20 + [1] * 5 + [0] * 10 + [0] * 15
    b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    assert cohen_kappa(a, b, categories=(0, 1)) == pytest.approx(0.4)
    # with two categories linear weights equal plain kappa
    assert cohen_kappa(a, b, categories=(0, 1), weights="linear") == pytest.approx(0.4)


def test_hand_computed_three_class_kappa() -> None:
    # O: (0,0), (1,2), (2,1) each 1/3; E uniform 1/9.
    # plain: 1 - (2/3)/(6/9) = 0 ; linear: 1 - (1/3)/(4/9) = 0.25
    a, b = [0, 1, 2], [0, 2, 1]
    assert cohen_kappa(a, b) == pytest.approx(0.0)
    assert cohen_kappa(a, b, weights="linear") == pytest.approx(0.25)


def test_perfect_agreement_is_one() -> None:
    a = [0, 1, 2, 2, 1, 0]
    assert cohen_kappa(a, a) == pytest.approx(1.0)
    assert cohen_kappa(a, a, weights="linear") == pytest.approx(1.0)


def test_linear_weights_penalize_far_disagreements_more() -> None:
    a = [0, 0, 1, 1, 2, 2, 0, 2]
    near = [0, 0, 1, 1, 2, 2, 1, 1]  # two off-by-one
    far = [0, 0, 1, 1, 2, 2, 2, 0]  # two off-by-two
    assert cohen_kappa(a, near, weights="linear") > cohen_kappa(a, far, weights="linear")


def test_kappa_undefined_cases_are_nan() -> None:
    assert math.isnan(cohen_kappa([], []))
    assert math.isnan(cohen_kappa([2, 2, 2], [2, 2, 2]))  # chance agreement is total


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="length"):
        cohen_kappa([0, 1], [0])
    with pytest.raises(ValueError, match="outside"):
        cohen_kappa([0, 3], [0, 1])
    with pytest.raises(ValueError, match="weights"):
        cohen_kappa([0], [0], weights="quadratic")  # type: ignore[arg-type]


def test_confusion_matrix_rows_are_a_cols_are_b() -> None:
    m = confusion_matrix([0, 0, 2], [0, 1, 2])
    assert m.tolist() == [[1, 1, 0], [0, 0, 0], [0, 0, 1]]


def test_agreement_report_uses_shared_graded_pairs() -> None:
    a = [
        j("q1", "x", 2),
        j("q1", "y", 0),
        j("q1", "z", None),  # unsure on A -> excluded
        j("q2", "x", 1),
        j("q2", "w", 2),  # B never judged -> only_a
        j("q1", "y", 1),  # A corrected y: last wins
    ]
    b = [
        j("q1", "x", 2, "B"),
        j("q1", "y", 1, "B"),
        j("q1", "z", 2, "B"),
        j("q2", "x", 0, "B"),
        j("q3", "v", 0, "B"),  # not in A -> only_b
    ]
    r = agreement_report(a, b)
    assert r.pairs == 3
    assert r.unsure_excluded == 1
    assert r.only_a == 1 and r.only_b == 1
    assert r.observed_agreement == pytest.approx(2 / 3)
    assert r.disagreements == [("q2", "x", 1, 0)]
    assert sum(map(sum, r.confusion)) == 3
    assert r.kappa == pytest.approx(cohen_kappa([2, 1, 1], [2, 1, 0]))
    assert r.as_dict()["disagreements"] == [{"qid": "q2", "image": "x", "a": 1, "b": 0}]


def test_empty_overlap_report() -> None:
    r = agreement_report([j("q1", "x", 1)], [])
    assert r.pairs == 0 and math.isnan(r.kappa) and math.isnan(r.observed_agreement)


def test_unresolved_and_final_qrels() -> None:
    a = [j("q1", "x", 2), j("q1", "y", 0), j("q2", "x", 1)]
    b = [j("q1", "x", 0, "B"), j("q1", "y", 1, "B")]
    report = agreement_report(a, b)
    assert unresolved(report, []) == [("q1", "x"), ("q1", "y")]

    adj = [j("q1", "x", 1, "adjudicated")]
    assert unresolved(report, adj) == [("q1", "y")]

    adj.append(j("q1", "y", None, "adjudicated"))  # resolved as unsure -> dropped
    assert unresolved(report, adj) == []
    assert final_qrels(a, adj) == {"q1": {"x": 1}, "q2": {"x": 1}}


def test_binarize_and_observed_agreement() -> None:
    assert binarize([0, 1, 2], 1) == [0, 1, 1]
    assert binarize([0, 1, 2], 2) == [0, 0, 1]
    assert observed_agreement([0, 1, 2], [0, 2, 2]) == pytest.approx(2 / 3)
    assert math.isnan(observed_agreement([], []))


def test_bootstrap_ci_is_seeded_and_brackets_value() -> None:
    a = [0, 1, 2, 2, 1, 0, 2, 1, 0, 2] * 3
    b = [0, 1, 2, 1, 1, 0, 2, 2, 0, 2] * 3
    stats = {"k": cohen_kappa, "po": observed_agreement}
    r1 = bootstrap_ci(a, b, stats, n_boot=200, seed=7)
    r2 = bootstrap_ci(a, b, stats, n_boot=200, seed=7)
    assert r1 == r2
    lo, hi = r1["k"]["ci"]
    assert lo <= cohen_kappa(a, b) <= hi


def test_bootstrap_clusters_resample_whole_groups() -> None:
    # Two clusters: one perfectly agreeing, one fully disagreeing -> some resamples
    # contain only one cluster, so po takes only the values 0, 0.5 or 1.
    a, b = [0, 0, 1, 1], [0, 0, 0, 0]
    clusters = ["q1", "q1", "q2", "q2"]
    r = bootstrap_ci(a, b, {"po": observed_agreement}, n_boot=100, seed=0, clusters=clusters)
    assert r["po"]["ci"][0] == pytest.approx(0.0)
    assert r["po"]["ci"][1] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="length"):
        bootstrap_ci(a, b, {"po": observed_agreement}, clusters=["q1"])


def test_bootstrap_counts_degenerate_resamples() -> None:
    r = bootstrap_ci([2, 2], [2, 2], {"k": cohen_kappa}, n_boot=10, seed=0)
    assert r["k"]["nan_resamples"] == 10 and math.isnan(r["k"]["ci"][0])


def _overlap_fixture() -> tuple[list[Judgment], list[Judgment], set[tuple[str, str]]]:
    a = [
        j("q1", "x", 2),
        j("q1", "y", 0),
        j("q1", "z", None),  # unsure A only
        j("q2", "x", 1),
        j("q2", "w", None),  # unsure both
        j("q2", "v", 2),
        j("q3", "x", 2),  # not in overlap -> ignored
    ]
    b = [
        j("q1", "x", 0, "B"),  # diff 2
        j("q1", "y", 0, "B"),
        j("q1", "z", 1, "B"),
        j("q2", "x", 2, "B"),  # diff 1
        j("q2", "w", None, "B"),
        j("q2", "v", None, "B"),  # unsure B only
    ]
    overlap = {("q1", "x"), ("q1", "y"), ("q1", "z"), ("q2", "x"), ("q2", "w"), ("q2", "v")}
    return a, b, overlap


def test_overlap_agreement_counts_and_ordering() -> None:
    a, b, overlap = _overlap_fixture()
    cats = {"q1": "skin", "q2": "lung"}
    d = overlap_agreement(a, b, overlap, cats, {"q1": "bcc"}, n_boot=50, seed=1)
    assert d["overlap_pairs"] == 6
    assert d["unsure"] == {
        "a": 2, "b": 2, "both": 1, "a_only": 1, "b_only": 1, "excluded_total": 3
    }  # fmt: skip
    assert d["comparable_pairs"] == 3 and d["comparable_queries"] == 2
    assert d["ordinal"]["observed_agreement"]["value"] == pytest.approx(1 / 3)
    assert d["ordinal"]["kappa"]["value"] == pytest.approx(cohen_kappa([2, 0, 1], [0, 0, 2]))
    assert [(x["qid"], x["abs_diff"]) for x in d["disagreements"]] == [("q1", 2), ("q2", 1)]
    assert d["disagreements"][0]["query"] == "bcc"
    # lenient: A [1,0,1] vs B [0,0,1]; strict: A [1,0,0] vs B [0,0,1]
    assert d["binary"]["lenient"]["observed_agreement"]["value"] == pytest.approx(2 / 3)
    assert d["binary"]["strict"]["observed_agreement"]["value"] == pytest.approx(1 / 3)
    assert "kappa_linear" not in d["binary"]["strict"]
    assert d["marginals_comparable"]["a"] == {"0": 1, "1": 1, "2": 1, "unsure": 0}
    assert d["marginals_overlap"]["b"] == {"0": 2, "1": 1, "2": 1, "unsure": 2}
    assert d["by_category"]["skin"] == {
        "pairs": 3, "unsure_excluded": 1, "comparable": 2, "agree": 1, "diff_1": 0, "diff_2": 1
    }  # fmt: skip
    assert d["bootstrap"]["seed"] == 1


def test_overlap_agreement_rejects_unjudged_or_uncategorized_pairs() -> None:
    a, b, overlap = _overlap_fixture()
    cats = {"q1": "skin", "q2": "lung"}
    with pytest.raises(ValueError, match="not judged by b"):
        overlap_agreement(a + [j("q9", "x", 1)], b, overlap | {("q9", "x")}, cats)
    with pytest.raises(ValueError, match="category"):
        overlap_agreement(a, b, overlap, {"q1": "skin"})


def test_markdown_and_limitations_render() -> None:
    a, b, overlap = _overlap_fixture()
    d = overlap_agreement(a, b, overlap, {"q1": "skin", "q2": "lung"}, n_boot=20, seed=0)
    lim = agreement_limitations(d, "A", "B", ["fixed fact"])
    assert lim[-1] == "fixed fact" and any("3 comparable pairs" in x for x in lim)
    doc = {
        "golden": "g",
        "command": "cmd",
        "git_sha": "abc",
        "annotators": {"a": "A", "b": "B"},
        **d,
        "limitations": lim,
    }
    md = render_agreement_markdown(doc)
    assert "| comparable pairs | 3 (2 queries) |" in md
    assert md.index("| 2 | q1 |") < md.index("| 1 | q2 |")
    assert "fixed fact" in md
