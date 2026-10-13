import math

import pytest

from pathosearch.eval.metrics import (
    average_precision_at_k,
    is_relevant,
    ndcg_at_k,
    precision_at_k,
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
