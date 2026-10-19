import numpy as np
import pytest

from pathosearch.eval.leakage_analysis import clusters, distribution, nearest_cosine, within_pairs


def test_nearest_cosine_cross() -> None:
    q = np.array([[1.0, 0.0], [0.0, 1.0]])
    r = np.array([[1.0, 0.0], [1.0, 1.0]])
    best, arg = nearest_cosine(q, r)
    assert arg.tolist() == [0, 1]
    assert best[0] == pytest.approx(1.0)
    assert best[1] == pytest.approx(2**-0.5, abs=1e-6)


def test_nearest_cosine_excludes_self() -> None:
    x = np.array([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0]])
    _, arg = nearest_cosine(x, x, exclude_self_rows=[0, 1, 2])
    assert arg.tolist() == [1, 0, 1]


def test_nearest_cosine_exclude_length_mismatch() -> None:
    x = np.eye(2)
    with pytest.raises(ValueError):
        nearest_cosine(x, x, exclude_self_rows=[0])


def test_distribution_counts() -> None:
    d = distribution(np.array([0.5, 0.96, 0.995, 1.0]))
    assert d["count_ge"] == {"0.95": 3, "0.99": 2, "0.999": 1, "0.9999": 1}
    assert d["max"] == 1.0
    assert sum(d["histogram"]["counts"]) == 4


def test_clusters_union_find() -> None:
    c = clusters([("a", "b"), ("b", "c"), ("x", "y")])
    assert c["n_images"] == 5
    assert c["n_clusters"] == 2
    assert c["largest_cluster_sizes"] == [3, 2]
    assert c["n_pairs"] == 3


def test_within_pairs() -> None:
    emb = np.array([[1.0, 0.0], [1.0, 0.01], [0.0, 1.0]])
    assert within_pairs(emb, ["a", "b", "c"], 0.95) == [("a", "b")]
