import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import pathosearch.encoders
from pathosearch.errors import DataNotFoundError
from pathosearch.eval import visualize_search
from pathosearch.faiss_search import _build_index, _save_index


class FakeEncoder:
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)


def _make_encoder_artifacts(art_dir: Path, ids: list[str]) -> None:
    art_dir.mkdir(parents=True)
    embs = np.eye(4, dtype=np.float32)[: len(ids)]
    np.save(art_dir / "embeddings.npy", embs)
    (art_dir / "ids.json").write_text(json.dumps(ids))
    _save_index(_build_index(embs, "ip"), str(art_dir), "ip", len(ids), 4)


@pytest.fixture
def corpus(tmp_path: Path) -> dict[str, Path]:
    ids = ["a.jpg", "b.jpg"]
    for enc in ("clip", "quilt"):
        _make_encoder_artifacts(tmp_path / "art" / enc, ids)
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for name in ids:
        Image.new("RGB", (8, 8), "pink").save(img_dir / name)
    caps = tmp_path / "captions.jsonl"
    caps.write_text(
        "\n".join(json.dumps({"path": n, "caption": f"cap {n}", "pathology": "Skin"}) for n in ids),
        encoding="utf-8",
    )
    return {"root": tmp_path, "images": img_dir, "caps": caps}


def _run(monkeypatch: pytest.MonkeyPatch, c: dict[str, Path], captions: Path) -> None:
    monkeypatch.setattr(pathosearch.encoders, "get_encoder", lambda name: FakeEncoder())
    argv = [
        "visualize_search",
        "--query", "skin",
        "--topk", "2",
        "--img_dir", str(c["images"]),
        "--clip_dir", str(c["root"] / "art" / "clip"),
        "--quilt_dir", str(c["root"] / "art" / "quilt"),
        "--captions", str(captions),
        "--out_dir", str(c["root"] / "reports"),
    ]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    visualize_search.main()


def test_visualize_search_writes_report(
    monkeypatch: pytest.MonkeyPatch, corpus: dict[str, Path]
) -> None:
    _run(monkeypatch, corpus, corpus["caps"])

    reports = list((corpus["root"] / "reports").glob("search_skin_*.html"))
    assert len(reports) == 1
    html = reports[0].read_text(encoding="utf-8")
    assert "skin" in html
    assert "data:image/jpeg;base64" in html  # images were found and embedded


def test_visualize_search_fails_loudly_without_captions(
    monkeypatch: pytest.MonkeyPatch, corpus: dict[str, Path]
) -> None:
    with pytest.raises(DataNotFoundError):
        _run(monkeypatch, corpus, corpus["root"] / "missing.jsonl")
