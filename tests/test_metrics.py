import math

import pytest

from pathosearch.eval.metrics import (
    average_precision_at_k,
    graded_ndcg_at_k,
    hit_at_k,
    is_relevant,
    ndcg_at_k,
    precision_at_k,
    precision_at_k_qrels,
    recall_at_k,
    reciprocal_rank,
)


def res(*labels: str) -> list[dict]:
    return [{"pathology": label} for label in labels]


def test_is_relevant() -> None:
    assert is_relevant({"pathology": "Renal"}, ["Renal", "Pulmonary"])
    assert not is_relevant({"pathology": "Skin"}, ["Renal"])
    assert not is_relevant({"pathology": "Renal"}, [])  # no expected categories -> never relevant


def test_precision_at_k() -> None:
    r = res("A", "B", "A", "C")
    assert precision_at_k(r, ["A"], 4) == pytest.approx(0.5)
    assert precision_at_k(r, ["A"], 2) == pytest.approx(0.5)  # only first two count
    assert precision_at_k(r, ["A", "B"], 3) == pytest.approx(1.0)
    assert precision_at_k(r, ["Z"], 4) == 0.0


def test_precision_divides_by_k_even_with_fewer_results() -> None:
    assert precision_at_k(res("A", "A"), ["A"], 10) == pytest.approx(0.2)


def test_average_precision_at_k() -> None:
    # hits at ranks 1 and 3 -> (1/1 + 2/3) / 2
    assert average_precision_at_k(res("A", "B", "A"), ["A"], 3) == pytest.approx((1 + 2 / 3) / 2)
    assert average_precision_at_k(res("B", "B", "B"), ["A"], 3) == 0.0
    assert average_precision_at_k(res("A", "A", "A"), ["A"], 3) == pytest.approx(1.0)
    # relevant item beyond k is ignored
    assert average_precision_at_k(res("B", "B", "A"), ["A"], 2) == 0.0


def test_ndcg_at_k() -> None:
    idcg3 = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(res("A", "A", "A"), ["A"], 3) == pytest.approx(1.0)
    assert ndcg_at_k(res("B", "B", "B"), ["A"], 3) == 0.0
    # relevant at ranks 1 and 3: dcg = 1 + 1/log2(4) = 1.5
    assert ndcg_at_k(res("A", "B", "A"), ["A"], 3) == pytest.approx(1.5 / idcg3)


def test_ndcg_ideal_dcg_assumes_k_relevant_docs_exist() -> None:
    # Documents the known property (STATUS §4 risk 7): one relevant doc at rank 1 cannot reach 1.0.
    idcg3 = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(res("A", "B", "B"), ["A"], 3) == pytest.approx(1 / idcg3)


def test_ndcg_with_fewer_results_than_k() -> None:
    assert ndcg_at_k(res("A"), ["A"], 3) == pytest.approx(
        1 / (1 + 1 / math.log2(3) + 1 / math.log2(4))
    )


@pytest.mark.parametrize("fn", [precision_at_k, average_precision_at_k, ndcg_at_k])
def test_empty_expected_gives_nan(fn) -> None:
    assert math.isnan(fn(res("A"), [], 3))


# --- qrels-based metrics ----------------------------------------------------

QRELS = {"a": 2, "b": 1, "c": 0, "d": 2}  # strict relevant: a, d; lenient: a, b, d


@pytest.mark.parametrize(
    ("ranked", "k", "min_rel", "expected"),
    [
        (["c", "a", "x", "b"], 4, 1, 0.5),
        (["c", "a", "x", "b"], 4, 2, 0.25),
        (["a"], 10, 1, 0.1),  # divides by k even with fewer results
    ],
)
def test_precision_at_k_qrels(ranked: list[str], k: int, min_rel: int, expected: float) -> None:
    assert precision_at_k_qrels(ranked, QRELS, k, min_rel) == pytest.approx(expected)


def test_hit_at_k() -> None:
    assert hit_at_k(["c", "x", "b"], QRELS, 3) == 1.0
    assert hit_at_k(["c", "x", "b"], QRELS, 3, min_rel=2) == 0.0
    assert hit_at_k(["c", "x", "b"], QRELS, 2) == 0.0  # b is beyond k
    assert hit_at_k([], QRELS, 5) == 0.0


def test_recall_at_k() -> None:
    assert recall_at_k(["a", "b", "x"], QRELS, 3) == pytest.approx(2 / 3)
    assert recall_at_k(["a", "b", "x"], QRELS, 3, min_rel=2) == pytest.approx(1 / 2)
    assert recall_at_k(["a", "b", "d"], QRELS, 2) == pytest.approx(2 / 3)
    assert recall_at_k(["a", "b", "d", "c"], QRELS, 10) == pytest.approx(1.0)


def test_unjudged_ids_count_as_non_relevant() -> None:
    assert recall_at_k(["x", "y", "z"], QRELS, 3) == 0.0
    assert reciprocal_rank(["x", "y", "a"], QRELS) == pytest.approx(1 / 3)


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["a", "b"], QRELS) == 1.0
    assert reciprocal_rank(["c", "x", "b"], QRELS) == pytest.approx(1 / 3)
    assert reciprocal_rank(["c", "x", "b"], QRELS, min_rel=2) == 0.0
    assert reciprocal_rank(["c", "x", "b"], QRELS, k=2) == 0.0  # cutoff applies


def test_graded_ndcg_perfect_ranking_is_one() -> None:
    assert graded_ndcg_at_k(["a", "d", "b", "c"], QRELS, 4) == pytest.approx(1.0)
    assert graded_ndcg_at_k(["a", "d"], QRELS, 2) == pytest.approx(1.0)


def test_graded_ndcg_hand_computed() -> None:
    # ranked b(1), x(0), a(2): dcg = 1/log2(2) + 3/log2(4) = 1 + 1.5
    # ideal a, d, b: 3 + 3/log2(3) + 1/log2(4)
    idcg = 3 + 3 / math.log2(3) + 0.5
    assert graded_ndcg_at_k(["b", "x", "a"], QRELS, 3) == pytest.approx(2.5 / idcg)


def test_graded_ndcg_strict_ignores_grade_one() -> None:
    # min_rel=2: b has no gain; ranked b, a -> dcg = 3/log2(3); ideal a, d -> 3 + 3/log2(3)
    expected = (3 / math.log2(3)) / (3 + 3 / math.log2(3))
    assert graded_ndcg_at_k(["b", "a"], QRELS, 2, min_rel=2) == pytest.approx(expected)


def test_graded_ndcg_few_relevant_docs_can_reach_one() -> None:
    # Contrast with the legacy ndcg_at_k (IDCG assumes k relevant docs).
    assert graded_ndcg_at_k(["a", "x", "y"], {"a": 2}, 3) == pytest.approx(1.0)


@pytest.mark.parametrize("fn", [precision_at_k_qrels, hit_at_k, recall_at_k, graded_ndcg_at_k])
def test_qrels_without_relevant_docs_give_nan(fn) -> None:
    assert math.isnan(fn(["a"], {}, 3))
    assert math.isnan(fn(["a"], {"a": 0}, 3))
    assert math.isnan(fn(["a"], {"a": 1}, 3, min_rel=2))


def test_reciprocal_rank_without_relevant_docs_is_nan() -> None:
    assert math.isnan(reciprocal_rank(["a"], {"a": 0}))


@pytest.mark.parametrize("fn", [precision_at_k_qrels, hit_at_k, recall_at_k, graded_ndcg_at_k])
def test_duplicate_ranked_ids_raise(fn) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        fn(["a", "a"], QRELS, 2)
