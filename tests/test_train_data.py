from pathlib import Path

import pytest
import torch

from conftest import make_image_files
from pathosearch.errors import DataNotFoundError
from pathosearch.train.data import (
    ClipCollator,
    PairDataset,
    epoch_order,
    load_captions_jsonl,
    select_scale,
    write_captions_jsonl,
)
from pathosearch.train.manifest import caption_for


def rows(n: int) -> list[dict]:
    # scale_rank deliberately not in file order
    return [{"image": f"img{i}.jpg", "scale_rank": (i * 7) % n} for i in range(n)]


def test_select_scale_is_rank_prefix_and_nested() -> None:
    r = rows(10)
    small, big = select_scale(r, 3), select_scale(r, 6)
    by_rank = {x["scale_rank"]: x["image"] for x in r}
    assert small == [by_rank[k] for k in range(3)]
    assert big[:3] == small
    with pytest.raises(ValueError, match="only 10"):
        select_scale(r, 11)


def test_select_scale_rejects_broken_ranks() -> None:
    r = rows(4)
    r[1]["scale_rank"] = r[0]["scale_rank"]
    with pytest.raises(ValueError, match="scale_rank"):
        select_scale(r, 2)


def test_captions_jsonl_round_trip(tmp_path: Path) -> None:
    caps = {"a.jpg": ["one", "two"], "b.jpg": ["x"]}
    write_captions_jsonl(tmp_path / "c.jsonl", caps)
    assert load_captions_jsonl(tmp_path / "c.jsonl") == caps


def test_pair_dataset_rotates_captions_per_epoch(tmp_path: Path) -> None:
    paths = make_image_files(tmp_path, ["a.jpg", "b.jpg"])
    caps = {"a.jpg": ["a0", "a1", "a2"], "b.jpg": ["b0"]}
    ds = PairDataset(["a.jpg", "b.jpg"], caps, paths, caption_seed=42)
    seen = []
    for e in range(3):
        ds.set_epoch(e)
        item = ds[0]
        assert item.name == "a.jpg" and item.image.mode == "RGB"
        assert item.caption == caption_for("a.jpg", caps["a.jpg"], e, seed=42)
        seen.append(item.caption)
    assert sorted(seen) == ["a0", "a1", "a2"]
    assert ds[1].caption == "b0" and len(ds) == 2


def test_pair_dataset_never_substitutes(tmp_path: Path) -> None:
    paths = make_image_files(tmp_path, ["a.jpg"])
    with pytest.raises(ValueError, match="no captions"):
        PairDataset(["a.jpg"], {"a.jpg": []}, paths)
    with pytest.raises(ValueError, match="no captions"):
        PairDataset(["a.jpg", "z.jpg"], {"a.jpg": ["c"]}, paths)
    ds = PairDataset(["z.jpg"], {"z.jpg": ["c"]}, paths)
    with pytest.raises(DataNotFoundError):
        ds[0]


def test_epoch_order_is_a_seeded_permutation() -> None:
    a = epoch_order(20, seed=0, epoch=0)
    assert sorted(a) == list(range(20))
    assert a == epoch_order(20, seed=0, epoch=0)
    assert a != epoch_order(20, seed=0, epoch=1)
    assert a != epoch_order(20, seed=1, epoch=0)


class FakeTokenizer:
    def __call__(self, texts, **kw):
        return {"input_ids": [[0] + [2] * len(t.split()) + [1] for t in texts]}


class FakeProcessor:
    tokenizer = FakeTokenizer()

    def __init__(self) -> None:
        self.kw: dict = {}

    def __call__(self, *, images, text, **kw):
        self.kw = kw
        n, L = len(images), kw["max_length"]
        return {
            "pixel_values": torch.zeros(n, 3, 4, 4),
            "input_ids": torch.zeros(n, L, dtype=torch.long),
            "attention_mask": torch.ones(n, L, dtype=torch.long),
        }


def test_clip_collator_counts_truncation(tmp_path: Path) -> None:
    paths = make_image_files(tmp_path, ["a.jpg", "b.jpg"])
    ds = PairDataset(["a.jpg", "b.jpg"], {"a.jpg": ["w " * 10], "b.jpg": ["w"]}, paths)
    proc = FakeProcessor()
    batch = ClipCollator(proc, max_length=5)([ds[0], ds[1]])
    assert batch.names == ["a.jpg", "b.jpg"]
    assert batch.truncated == 1  # 12 tokens > 5, 3 tokens fit
    assert proc.kw["truncation"] is True and proc.kw["padding"] == "max_length"
    assert batch.pixel_values.shape[0] == 2 and batch.input_ids.shape == (2, 5)
