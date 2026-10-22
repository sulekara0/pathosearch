import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from conftest import make_image_files
from pathosearch.errors import DataNotFoundError
from pathosearch.train.bundle import build_bundle, resize_short_side
from pathosearch.train.data import load_captions_jsonl


def test_resize_short_side_keeps_aspect_and_never_upscales() -> None:
    assert resize_short_side(Image.new("RGB", (600, 400)), 224).size == (336, 224)
    assert resize_short_side(Image.new("RGB", (300, 900)), 224).size == (224, 672)
    small = Image.new("RGB", (100, 150))
    assert resize_short_side(small, 224).size == (100, 150)


def test_build_bundle(tmp_path: Path) -> None:
    names = ["b.jpg", "a.jpg", "c.png"]
    paths = make_image_files(tmp_path / "src", names)
    caps = {n: [f"cap {n}"] for n in names}
    out = tmp_path / "bundle.zip"
    meta = build_bundle(names, paths, caps, out, size=16, quality=90)
    assert meta["images"] == 3 and meta["size"] == 16 and len(meta["zip_sha256"]) == 64
    with zipfile.ZipFile(out) as z:
        listed = sorted(z.namelist())
        assert listed == ["captions.jsonl", "images/a.jpg", "images/b.jpg", "images/c.png"]
        with z.open("images/a.jpg") as fh:
            assert min(Image.open(fh).size) == 16
        z.extract("captions.jsonl", tmp_path / "x")
    assert load_captions_jsonl(tmp_path / "x" / "captions.jsonl") == caps
    assert json.loads(json.dumps(meta)) == meta


def test_build_bundle_refuses_missing_and_overwrite(tmp_path: Path) -> None:
    paths = make_image_files(tmp_path / "src", ["a.jpg"])
    with pytest.raises(DataNotFoundError):
        build_bundle(
            ["a.jpg", "z.jpg"], paths, {"a.jpg": ["c"], "z.jpg": ["c"]}, tmp_path / "o.zip"
        )
    with pytest.raises(ValueError, match="no captions"):
        build_bundle(["a.jpg"], paths, {}, tmp_path / "o2.zip")
    out = tmp_path / "o3.zip"
    build_bundle(["a.jpg"], paths, {"a.jpg": ["c"]}, out)
    with pytest.raises(FileExistsError):
        build_bundle(["a.jpg"], paths, {"a.jpg": ["c"]}, out)
