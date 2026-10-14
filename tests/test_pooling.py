from collections import Counter
from pathlib import Path

import pytest

from pathosearch.eval.pooling import (
    SOURCE_POOLED,
    SOURCE_RANDOM,
    build_pool,
    load_pool,
    overlap_subset,
    pool_stats,
    save_pool,
)

CORPUS = [f"img{i:03d}.jpg" for i in range(100)]


def run(*indices: int) -> list[str]:
    return [CORPUS[i] for i in indices]


RUNS = {
    "clip": {"q1": run(0, 1, 2, 3), "q2": run(10, 11, 12, 13)},
    "quilt": {"q1": run(2, 3, 4, 5), "q2": run(10, 20, 21, 22)},
}


def items_for(pool, qid):
    return [it for it in pool if it.qid == qid]


def test_pool_is_union_of_top_depth_plus_random_controls() -> None:
    pool = build_pool(RUNS, CORPUS, depth=3, n_random=2)
    q1 = items_for(pool, "q1")
    pooled = {it.image for it in q1 if it.source == SOURCE_POOLED}
    assert pooled == set(run(0, 1, 2, 3, 4))  # depth 3 cuts img003 from clip, img005 from quilt
    randoms = [it for it in q1 if it.source == SOURCE_RANDOM]
    assert len(randoms) == 2
    assert not {it.image for it in randoms} & pooled
    assert all(it.systems == () for it in randoms)


def test_systems_record_who_retrieved_each_image() -> None:
    pool = build_pool(RUNS, CORPUS, depth=3, n_random=0)
    by_image = {it.image: it.systems for it in items_for(pool, "q1")}
    assert by_image[CORPUS[2]] == ("clip", "quilt")
    assert by_image[CORPUS[0]] == ("clip",)
    assert by_image[CORPUS[4]] == ("quilt",)


def test_order_is_a_shuffled_permutation_per_query() -> None:
    pool = build_pool(RUNS, CORPUS, depth=4, n_random=5)
    for qid in ("q1", "q2"):
        orders = sorted(it.order for it in items_for(pool, qid))
        assert orders == list(range(len(orders)))
    # display order must not reveal pooled-vs-random (randoms not all at the end)
    q1 = sorted(items_for(pool, "q1"), key=lambda it: it.order)
    assert [it.source for it in q1] != sorted(it.source for it in q1)


def test_pool_is_deterministic() -> None:
    assert build_pool(RUNS, CORPUS, seed=1) == build_pool(RUNS, CORPUS, seed=1)
    assert build_pool(RUNS, CORPUS, seed=1) != build_pool(RUNS, CORPUS, seed=2)


def test_adding_queries_keeps_existing_pool_unchanged() -> None:
    small = {s: {"q1": r["q1"]} for s, r in RUNS.items()}
    before = build_pool(small, CORPUS)
    after = items_for(build_pool(RUNS, CORPUS), "q1")
    assert before == after


def test_random_controls_capped_by_available_images() -> None:
    corpus = run(0, 1, 2, 3, 4, 5)
    pool = build_pool(RUNS | {}, corpus + run(10, 11, 12, 13, 20, 21, 22), depth=4, n_random=50)
    q1 = items_for(pool, "q1")
    assert len(q1) == 13  # every corpus image, none twice
    assert len({it.image for it in q1}) == 13


def test_no_random_slots_left_when_pool_covers_corpus() -> None:
    corpus = run(0, 1)
    pool = build_pool({"s": {"q": corpus}}, corpus, depth=5, n_random=3)
    assert [it.source for it in pool] == [SOURCE_POOLED, SOURCE_POOLED]


def test_unknown_image_in_run_is_rejected() -> None:
    with pytest.raises(ValueError, match="not in corpus"):
        build_pool({"s": {"q": ["ghost.jpg"]}}, CORPUS)


def test_system_missing_a_query_is_rejected() -> None:
    runs = {"a": {"q1": run(0)}, "b": {"q2": run(1)}}
    with pytest.raises(ValueError, match="no run"):
        build_pool(runs, CORPUS)


@pytest.mark.parametrize(("depth", "n_random"), [(0, 5), (10, -1)])
def test_invalid_parameters(depth: int, n_random: int) -> None:
    with pytest.raises(ValueError):
        build_pool(RUNS, CORPUS, depth=depth, n_random=n_random)


def test_overlap_subset_is_per_query_and_at_least_one() -> None:
    pool = build_pool(RUNS, CORPUS, depth=4, n_random=5)
    subset = overlap_subset(pool, frac=0.2)
    sizes = Counter(it.qid for it in subset)
    full = Counter(it.qid for it in pool)
    for qid in full:
        assert sizes[qid] == max(1, round(full[qid] * 0.2))
    assert set(subset) <= set(pool)
    assert overlap_subset(pool, frac=0.01) and all(
        n == 1 for n in Counter(it.qid for it in overlap_subset(pool, frac=0.01)).values()
    )


def test_overlap_subset_deterministic_and_stable_when_growing() -> None:
    pool = build_pool(RUNS, CORPUS)
    assert overlap_subset(pool) == overlap_subset(pool)
    q1_only = [it for it in pool if it.qid == "q1"]
    assert overlap_subset(q1_only) == [it for it in overlap_subset(pool) if it.qid == "q1"]


def test_overlap_invalid_frac() -> None:
    with pytest.raises(ValueError):
        overlap_subset([], frac=0.0)


def test_pool_stats() -> None:
    pool = build_pool(RUNS, CORPUS, depth=3, n_random=2)
    stats = pool_stats(pool)
    assert stats["queries"] == 2
    assert stats["pairs"] == len(pool)
    assert stats["by_source"][SOURCE_RANDOM] == 4
    # depth 3 -> q1: clip-only 0, 1; quilt-only 3, 4. q2: clip-only 11, 12; quilt-only 20, 21
    assert stats["unique_to_one_system"] == {"clip": 4, "quilt": 4}


def test_pool_round_trip_and_duplicate_rejected(tmp_path: Path) -> None:
    path = tmp_path / "pool.jsonl"
    pool = build_pool(RUNS, CORPUS)
    save_pool(path, pool)
    assert load_pool(path) == pool
    save_pool(path, [pool[0], pool[0]])
    with pytest.raises(ValueError, match="duplicate"):
        load_pool(path)
