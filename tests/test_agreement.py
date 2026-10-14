import math

import pytest

from pathosearch.eval.agreement import (
    agreement_report,
    cohen_kappa,
    confusion_matrix,
    final_qrels,
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
