import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pathosearch.data.sharded_embeddings import (
    benchmark,
    encode_names,
    make_shards,
    merge_shards,
    progress,
    run_shards,
    shard_paths,
)


class FakeEncoder:
    """Embedding = mean RGB of the image; counts calls."""

    def __init__(self) -> None:
        self.seen = 0

    def encode_images(self, images: list) -> np.ndarray:
        self.seen += len(images)
        return np.array([np.asarray(im, np.float32).mean(axis=(0, 1)) for im in images])


def make_images(tmp_path: Path, n: int) -> dict[str, Path]:
    paths = {}
    for i in range(n):
        p = tmp_path / f"img_{i:02d}.jpg"
        Image.new("RGB", (8, 8), (i, 2 * i, 3 * i)).save(p, quality=100)
        paths[p.name] = p
    return paths


def test_make_shards_sorted_unique() -> None:
    assert make_shards(["c", "a", "b", "a", "d"], 2) == [["a", "b"], ["c", "d"]]
    with pytest.raises(ValueError):
        make_shards(["a"], 0)


def test_encode_names_reports_unreadable_and_missing(tmp_path: Path) -> None:
    paths = make_images(tmp_path, 3)
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    paths["bad.jpg"] = bad
    emb, ids, failed = encode_names(
        ["img_00.jpg", "bad.jpg", "img_02.jpg", "gone.jpg"], paths, FakeEncoder(), 1
    )
    assert ids == ["img_00.jpg", "img_02.jpg"]
    assert failed == ["bad.jpg", "gone.jpg"]
    assert emb.shape == (2, 3)


def test_run_resume_and_merge(tmp_path: Path) -> None:
    paths = make_images(tmp_path, 5)
    out = tmp_path / "out"
    shards = make_shards(list(paths), 2)  # 3 shards: 2 + 2 + 1
    enc = FakeEncoder()
    assert run_shards(shards, paths, enc, out, batch_size=2, log=lambda _: None) == {
        "shards": 3,
        "written": 3,
        "skipped": 0,
    }
    assert enc.seen == 5

    # simulate an interruption after shard 1's .npy was written but before its .json
    shard_paths(out, 1)[1].unlink()
    assert progress(shards, out) == (2, 3)
    enc2 = FakeEncoder()
    stats = run_shards(shards, paths, enc2, out, batch_size=2, log=lambda _: None)
    assert stats == {"shards": 3, "written": 1, "skipped": 2}
    assert enc2.seen == 2

    info = merge_shards(shards, out, {"encoder": "fake"})
    ids = json.loads((out / "ids.json").read_text())
    emb = np.load(out / "embeddings.npy")
    assert ids == sorted(paths)
    assert emb.shape == (5, 3)
    assert np.allclose(emb[1], FakeEncoder().encode_images([Image.open(paths[ids[1]])])[0])
    assert info["n_images"] == 5 and info["n_failed"] == 0 and info["encoder"] == "fake"
    assert not list(out.glob("*.tmp"))


def test_merge_refuses_missing_or_foreign_shard(tmp_path: Path) -> None:
    paths = make_images(tmp_path, 4)
    out = tmp_path / "out"
    shards = make_shards(list(paths), 2)
    run_shards(shards, paths, FakeEncoder(), out, log=lambda _: None)
    with pytest.raises(ValueError, match="different name list"):
        merge_shards(make_shards(list(paths)[:3], 2), out, {})
    shard_paths(out, 0)[0].unlink()
    with pytest.raises(FileNotFoundError):
        merge_shards(shards, out, {})


def test_benchmark_estimate(tmp_path: Path) -> None:
    paths = make_images(tmp_path, 4)
    r = benchmark(list(paths), paths, FakeEncoder(), n=2, total=1000)
    assert r["sample"] == 2 and r["encoded"] == 2 and r["total_images"] == 1000
    assert r["estimated_total_minutes"] >= 0
