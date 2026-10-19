import numpy as np
import pytest

from pathosearch.eval.dedup import dedup_within, exclude_near_dup


def test_exclude_near_dup_any_encoder() -> None:
    g = {"clip": np.array([[1.0, 0.0], [0.0, 1.0]]), "quilt": np.array([[1.0, 0.0], [0.0, 1.0]])}
    m = {"clip": np.array([[1.0, 0.0], [0.7, 0.7]]), "quilt": np.array([[0.0, 1.0], [1.0, 1.0]])}
    out = exclude_near_dup(g, ["a", "b"], m, ["x", "y"], 0.99)
    # a matches x in clip; b matches x in quilt: each flagged by one encoder only
    assert [r["image"] for r in out] == ["a", "b"]
    assert list(out[0]["encoders"]) == ["clip"] and list(out[1]["encoders"]) == ["quilt"]
    assert out[0]["encoders"]["clip"]["nearest"] == "x"


def test_exclude_near_dup_none_below_tau() -> None:
    g = {"clip": np.array([[1.0, 0.0]])}
    m = {"clip": np.array([[0.0, 1.0]])}
    assert exclude_near_dup(g, ["a"], m, ["x"], 0.99) == []


def test_exclude_near_dup_encoder_mismatch() -> None:
    with pytest.raises(ValueError):
        exclude_near_dup({"clip": np.eye(2)}, ["a", "b"], {"quilt": np.eye(2)}, ["x", "y"], 0.9)


def test_dedup_within_keeps_smallest_name_per_cluster() -> None:
    e = np.array([[1.0, 0.0], [1.0, 0.001], [0.0, 1.0], [0.001, 1.0], [0.7, 0.7]])
    ids = ["d", "c", "b", "a", "e"]
    kept, dropped = dedup_within({"clip": e}, ids, 0.99)
    assert kept == ["c", "a", "e"]
    assert {(d["image"], d["representative"]) for d in dropped} == {("d", "c"), ("b", "a")}


def test_dedup_within_any_encoder_links() -> None:
    clip = np.array([[1.0, 0.0], [0.0, 1.0]])
    quilt = np.array([[1.0, 0.0], [1.0, 0.0]])
    kept, _ = dedup_within({"clip": clip, "quilt": quilt}, ["a", "b"], 0.99)
    assert kept == ["a"]


def test_dedup_within_transitive_chain() -> None:
    e = np.array([[1.0, 0.0], [0.97, 0.243], [0.88, 0.475]])  # a-b, b-c linked, a-c not
    kept, _ = dedup_within({"clip": e}, ["a", "b", "c"], 0.96)
    assert kept == ["a"]
