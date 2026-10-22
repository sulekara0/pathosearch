import numpy as np
import pytest

from pathosearch.eval.heldout import LeakageError
from pathosearch.train.manifest import (
    STEPS,
    ImageMeta,
    caption_for,
    collect_meta,
    drop_val_near_train,
    exclude,
    make_split,
    normalize_caption,
    render_markdown,
    scale_order,
    scale_sizes,
    split_val_by_group,
    summary,
    train_rows,
)

GOLD = ["vidGGGGGGGG_image_g.jpg", "900_0.jpg"]


def unit(*v: float) -> np.ndarray:
    a = np.array(v, np.float32)
    return a / np.linalg.norm(a)


def fixture():
    """Candidates engineered so that each exclusion step removes exactly one image."""
    names = [
        "noemb.jpg",  # no_embedding
        "near.jpg",  # near_dup_golden (cosine 1.0 to golden 0 in quilt only)
        "nearest.jpg",  # golden_nearest
        "vidGGGGGGGG_image_t.jpg",  # source_group (same video as golden 0)
        "samecap.jpg",  # caption_equal
        "nocap.jpg",  # no_caption
        "keep_a.jpg",
        "keep_b.jpg",
    ]
    emb_ids = [n for n in names if n != "noemb.jpg"] + ["unrelated.jpg"]
    rng = np.random.default_rng(0)
    clip = rng.normal(size=(len(emb_ids), 4)).astype(np.float32)
    quilt = rng.normal(size=(len(emb_ids), 4)).astype(np.float32)
    g_clip = np.stack([unit(1, 0, 0, 0), unit(0, 1, 0, 0)])
    g_quilt = np.stack([unit(0, 0, 1, 0), unit(0, 0, 0, 1)])
    # keep everything else far from the golden vectors in both encoders
    clip[:, :2] = -np.abs(clip[:, :2])
    quilt[:, 2:] = -np.abs(quilt[:, 2:])
    quilt[emb_ids.index("near.jpg")] = g_quilt[0]
    meta = {n: ImageMeta("bone", "quilt", [f"caption of {n}"]) for n in names + GOLD}
    meta["samecap.jpg"].captions = ["other", "  The GOLDEN   caption "]
    meta["900_0.jpg"].captions = ["the golden caption"]
    meta["nocap.jpg"].captions = []
    return names, emb_ids, {"clip": clip, "quilt": quilt}, {"clip": g_clip, "quilt": g_quilt}, meta


def run_exclude():
    names, emb_ids, emb, g_emb, meta = fixture()
    return exclude(names, emb_ids, GOLD, ["nearest.jpg"], emb, emb_ids, g_emb, meta), meta


def test_exclusion_steps_in_order_with_counts() -> None:
    excl, _ = run_exclude()
    assert [s["step"] for s in excl.steps] == list(STEPS)
    removed = {s["step"]: (s["removed"], s["examples"]) for s in excl.steps}
    assert removed == {
        "no_embedding": (1, ["noemb.jpg"]),
        "golden_ids": (0, []),
        "golden_nearest": (1, ["nearest.jpg"]),
        "near_dup_golden": (1, ["near.jpg"]),
        "source_group": (1, ["vidGGGGGGGG_image_t.jpg"]),
        "caption_equal": (1, ["samecap.jpg"]),
        "no_caption": (1, ["nocap.jpg"]),
    }
    assert excl.kept == ["keep_a.jpg", "keep_b.jpg"]
    assert excl.steps[-1]["remaining"] == 2


def test_golden_candidate_in_candidates_is_removed_not_leaked() -> None:
    names, emb_ids, emb, g_emb, meta = fixture()
    excl = exclude(names + [GOLD[1]], emb_ids, GOLD, [], emb, emb_ids, g_emb, meta)
    assert GOLD[1] not in excl.kept


def test_missing_embedding_after_filter_raises() -> None:
    names, emb_ids, emb, g_emb, meta = fixture()
    with pytest.raises(ValueError, match="lack embeddings"):
        exclude(names, names, GOLD, [], emb, emb_ids, g_emb, meta)


def test_assert_disjoint_guard_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    import pathosearch.train.manifest as m

    def boom(train, golden):  # simulate a leak slipping through earlier steps
        raise LeakageError("leak")

    monkeypatch.setattr(m, "assert_disjoint", boom)
    with pytest.raises(LeakageError):
        run_exclude()


def test_caption_rotation_is_deterministic_and_covers_all() -> None:
    caps = ["a", "b", "c"]
    seq = [caption_for("img.jpg", caps, e) for e in range(6)]
    assert seq == [caption_for("img.jpg", caps, e) for e in range(6)]
    assert sorted(seq[:3]) == caps and seq[:3] == seq[3:]
    assert caption_for("img.jpg", ["only"], 7) == "only"
    with pytest.raises(ValueError):
        caption_for("img.jpg", [], 0)


def test_collect_meta_filters_like_prepare_quilt() -> None:
    rows = [
        {"image_path": "x.jpg", "caption": " first ", "pathology": "['Bone']", "not_histology": "0", "subset": "quilt"},
        {"image_path": "x.jpg", "caption": "first", "pathology": "['Renal']", "not_histology": "0.0", "subset": "quilt"},
        {"image_path": "x.jpg", "caption": "not histo", "pathology": None, "not_histology": "1", "subset": "quilt"},
        {"image_path": "x.jpg", "caption": "  ", "pathology": None, "not_histology": "0", "subset": "quilt"},
        {"image_path": "x.jpg", "caption": "second", "pathology": None, "not_histology": "0", "subset": "quilt"},
        {"image_path": "y.jpg", "caption": float("nan"), "pathology": float("nan"), "not_histology": "nan", "subset": "laion"},
    ]  # fmt: skip
    meta = collect_meta(rows)
    assert meta["x.jpg"].captions == ["first", "second"]
    assert meta["x.jpg"].category == "bone"  # first row's first pathology
    assert meta["y.jpg"].captions == [] and meta["y.jpg"].category == "unknown"


def test_normalize_caption() -> None:
    assert normalize_caption("  A  b\nC ") == "a b c"


def test_split_val_by_whole_groups() -> None:
    imgs = [f"vidAAAAAAAA_image_{i}.jpg" for i in range(5)] + [f"{i}_0.jpg" for i in range(10)]
    train, val, groups = split_val_by_group(imgs, val_size=3, seed=1)
    assert set(train) | set(val) == set(imgs) and not set(train) & set(val)
    assert len(val) >= 3
    # a video never straddles train and val
    in_val = {n.startswith("vidAAAAAAAA") for n in val}
    in_train = {n.startswith("vidAAAAAAAA") for n in train}
    assert not (True in in_val and True in in_train)
    assert split_val_by_group(imgs, 3, seed=1) == (train, val, groups)


def test_drop_val_near_train() -> None:
    ids = ["t1", "t2", "v1", "v2"]
    e = np.stack([unit(1, 0), unit(0, 1), unit(1, 0), unit(1, 1)])
    other = np.stack([unit(1, 0), unit(0, 1), unit(0, 1), unit(1, 1)])
    kept, dropped = drop_val_near_train(["v1", "v2"], ["t1", "t2"], {"a": e}, ids)
    assert kept == ["v2"] and dropped == ["v1"]
    # ANY encoder: v1 near t2 in the second encoder too, v2 still kept
    kept, dropped = drop_val_near_train(["v1", "v2"], ["t1", "t2"], {"a": e, "b": other}, ids)
    assert dropped == ["v1"]


def test_scales_are_nested_prefixes_and_checked() -> None:
    order = scale_order([f"{i}.jpg" for i in range(50)], seed=3)
    assert order == scale_order(reversed([f"{i}.jpg" for i in range(50)]), seed=3)
    assert order != scale_order([f"{i}.jpg" for i in range(50)], seed=4)
    assert scale_sizes(50, {"a": 10, "b": 30, "full": None}) == {"a": 10, "b": 30, "full": 50}
    with pytest.raises(ValueError, match="exceeds"):
        scale_sizes(20, {"a": 10, "b": 30})


def test_make_split_rows_and_summary() -> None:
    excl, meta = run_exclude()
    names, emb_ids, emb, _, _ = fixture()
    split = make_split(excl.kept, emb, emb_ids, val_size=1, seed=0)
    assert len(split.val) == 1 and len(split.train) == 1
    rows = train_rows(split, meta)
    assert rows[0]["scale_rank"] == 0 and rows[0]["n_captions"] == 1
    s = summary(excl, split, meta, n_candidates=len(names), scales={"full": None})
    assert s["after_exclusion"] == 2 and s["val"]["images"] == 1
    assert s["scales"] == {"full": 1} and s["candidates"] == len(names)
    with pytest.raises(ValueError, match="exceeds"):  # default 10k/30k do not fit 1 image
        summary(excl, split, meta, n_candidates=len(names))
    md = render_markdown(
        {**s, "name": "t", "command": "cmd", "git_sha": "abc", "seed": 0}
        | {"near_dup_tau": 0.99, "val_size": 1}
    )
    assert "| no_embedding | 1 | 7 |" in md and "full=1" in md
