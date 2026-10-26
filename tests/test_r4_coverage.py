import pytest

from pathosearch.eval.r4_coverage import (
    adapter_dirs,
    coverage_report,
    coverage_summary,
    unjudged_pairs,
)

RUNS = {
    "a": {"q1": ["i1", "i2", "i3"], "q2": ["i4", "i5", "i6"]},
    "b": {"q1": ["i1", "i7", "i3"], "q2": ["i4", "i5", "i8"]},
}
JUDGED = {"q1": {"i1", "i2"}, "q2": {"i4"}}


def test_unjudged_pairs_respects_k() -> None:
    out = unjudged_pairs(RUNS, JUDGED, k=2)
    assert out["a"] == {"q1": [], "q2": ["i5"]}
    assert out["b"] == {"q1": ["i7"], "q2": ["i5"]}


def test_coverage_summary_counts_and_union() -> None:
    s = coverage_summary(RUNS, JUDGED, k=3)
    assert s["systems"]["a"]["unjudged_pairs"] == 3  # i3, i5, i6
    assert s["systems"]["a"]["judged_at_k"] == 0.5
    assert s["systems"]["b"]["unjudged_by_query"] == {"q1": 2, "q2": 2}
    # union: (q1,i3) (q2,i5) (q2,i6) (q1,i7) (q2,i8)
    assert s["unjudged_union"] == 5


def test_coverage_report_per_split() -> None:
    rep = coverage_report({"dev": RUNS}, {"dev": JUDGED}, k=3)
    assert rep["k"] == 3 and set(rep["splits"]) == {"dev"}


def test_adapter_dirs_parses_and_rejects() -> None:
    assert adapter_dirs(["x=some/dir"])["x"].as_posix() == "some/dir"
    with pytest.raises(SystemExit):
        adapter_dirs(["nodir"])
