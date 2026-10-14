from collections import Counter
from pathlib import Path

import pytest

from pathosearch.eval.golden import GoldenQuery
from pathosearch.eval.splits import (
    assert_no_split_leakage,
    check_split_leakage,
    default_dev_frac,
    load_splits,
    save_splits,
    split_queries,
    token_jaccard,
)


def make_queries(per_category: dict[str, int]) -> list[GoldenQuery]:
    return [
        GoldenQuery(f"{cat[:3]}{i:02d}", f"{cat} query number {i}", cat, "test")
        for cat, n in per_category.items()
        for i in range(n)
    ]


def category_of(queries: list[GoldenQuery]) -> dict[str, str]:
    return {q.qid: q.category for q in queries}


@pytest.mark.parametrize(("n", "frac"), [(20, 0.5), (50, 0.5), (51, 0.3), (100, 0.3)])
def test_default_dev_frac(n: int, frac: float) -> None:
    assert default_dev_frac(n) == frac


def test_pilot_size_is_split_half_and_half_per_category() -> None:
    qs = make_queries({c: 4 for c in "abcde"})  # 20 queries, pilot shape
    splits = split_queries(qs)
    assert len(splits["dev"]) == 10 and len(splits["test"]) == 10
    cat = category_of(qs)
    assert Counter(cat[q] for q in splits["dev"]) == {c: 2 for c in "abcde"}


def test_large_set_uses_30_percent_dev_and_is_stratified() -> None:
    qs = make_queries({c: 20 for c in "abcde"})
    splits = split_queries(qs)
    assert len(splits["dev"]) == 30 and len(splits["test"]) == 70
    cat = category_of(qs)
    assert Counter(cat[q] for q in splits["dev"]) == {c: 6 for c in "abcde"}


def test_uneven_categories_hit_exact_total() -> None:
    qs = make_queries({"a": 7, "b": 3, "c": 1})
    splits = split_queries(qs, dev_frac=0.3)
    assert len(splits["dev"]) == round(11 * 0.3)
    assert set(splits["dev"]) | set(splits["test"]) == {q.qid for q in qs}


def test_split_is_deterministic_for_seed_and_changes_with_seed() -> None:
    qs = make_queries({c: 20 for c in "abcde"})
    assert split_queries(qs, seed=42) == split_queries(qs, seed=42)
    assert split_queries(qs, seed=42) != split_queries(qs, seed=7)


def test_split_does_not_depend_on_input_order() -> None:
    qs = make_queries({c: 10 for c in "abc"})
    assert split_queries(qs) == split_queries(list(reversed(qs)))


def test_forced_dev_qids_stay_in_dev_when_set_grows() -> None:
    pilot = make_queries({c: 4 for c in "abcde"})
    pilot_ids = {q.qid for q in pilot}
    grown = make_queries({c: 20 for c in "abcde"})  # same first 4 qids per category
    assert pilot_ids <= {q.qid for q in grown}
    splits = split_queries(grown, force_dev=pilot_ids)
    assert pilot_ids <= set(splits["dev"])
    assert len(splits["dev"]) == 30  # pilot (20) counts toward the quota


def test_forced_unknown_qid_rejected() -> None:
    with pytest.raises(ValueError, match="force_dev"):
        split_queries(make_queries({"a": 2}), force_dev=["nope"])


@pytest.mark.parametrize("frac", [0.0, 1.0, 1.5])
def test_invalid_dev_frac_rejected(frac: float) -> None:
    with pytest.raises(ValueError, match="dev_frac"):
        split_queries(make_queries({"a": 4}), dev_frac=frac)


def test_splits_round_trip(tmp_path: Path) -> None:
    qs = make_queries({c: 4 for c in "abcde"})
    splits = split_queries(qs)
    path = tmp_path / "splits.json"
    save_splits(path, splits, seed=42, dev_frac=0.5)
    assert load_splits(path) == splits


def test_load_splits_rejects_overlap(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    save_splits(path, {"dev": ["q1"], "test": ["q1"]}, seed=42, dev_frac=0.5)
    with pytest.raises(ValueError, match="both"):
        load_splits(path)


def test_token_jaccard() -> None:
    assert token_jaccard("Colon adenocarcinoma", "colon, ADENOCARCINOMA") == 1.0
    assert token_jaccard("colon adenocarcinoma", "lung adenocarcinoma") == pytest.approx(1 / 3)


def q(qid: str, text: str) -> GoldenQuery:
    return GoldenQuery(qid, text, "x", "test")


def test_cross_split_exact_duplicate_is_an_error() -> None:
    qs = [q("q1", "Melanoma histology"), q("q2", "melanoma, histology"), q("q3", "renal")]
    report = check_split_leakage(qs, {"dev": ["q1"], "test": ["q2", "q3"]})
    assert report.cross_split_exact_duplicates == [("q1", "q2")]
    with pytest.raises(ValueError, match="same query text"):
        assert_no_split_leakage(qs, {"dev": ["q1"], "test": ["q2", "q3"]})


def test_near_duplicates_are_warnings_only() -> None:
    qs = [
        q("q1", "basal cell carcinoma of the skin biopsy"),
        q("q2", "basal cell carcinoma of skin biopsy"),
        q("q3", "renal cell carcinoma"),
    ]
    report = assert_no_split_leakage(qs, {"dev": ["q1"], "test": ["q2", "q3"]})
    assert [(a, b, crosses) for a, b, _, crosses in report.near_duplicates] == [("q1", "q2", True)]


def test_unassigned_and_unknown_qids_are_errors() -> None:
    qs = [q("q1", "a"), q("q2", "b")]
    report = check_split_leakage(qs, {"dev": ["q1", "q9"], "test": []})
    assert report.unassigned == ["q2"]
    assert report.unknown_qids == ["q9"]
    assert len(report.errors) == 2


def test_generated_split_passes_leakage_check() -> None:
    qs = make_queries({c: 20 for c in "abcde"})
    assert not assert_no_split_leakage(qs, split_queries(qs)).errors
