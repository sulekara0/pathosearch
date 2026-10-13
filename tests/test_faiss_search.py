import json
from pathlib import Path

import faiss
import numpy as np

from pathosearch.faiss_search import _build_index, _save_index, load_embeddings, load_index


def _write_artifacts(art_dir: Path, n: int = 20, d: int = 8) -> np.ndarray:
    rng = np.random.default_rng(0)
    embs = rng.normal(size=(n, d)).astype(np.float32)
    art_dir.mkdir(parents=True)
    np.save(art_dir / "embeddings.npy", embs)
    (art_dir / "ids.json").write_text(json.dumps([f"img_{i}.jpg" for i in range(n)]))
    return embs


def test_index_roundtrip_in_non_ascii_directory(tmp_path: Path) -> None:
    art = tmp_path / "çağrı_ışık" / "quilt"  # non-ASCII path that broke the FAISS C API on Windows
    embs = _write_artifacts(art)
    _save_index(_build_index(embs, "ip"), str(art), metric="ip", n=len(embs), d=embs.shape[1])

    index, metric = load_index(str(art))
    _, ids = load_embeddings(str(art))

    assert metric == "ip"
    assert index.ntotal == len(ids) == 20
    q = embs[3:4] / np.linalg.norm(embs[3:4])
    _, nn = index.search(q.astype(np.float32), 1)
    assert ids[nn[0][0]] == "img_3.jpg"


def test_load_index_reads_files_written_by_faiss_write_index(tmp_path: Path) -> None:
    # Backward compatibility: indexes built before this change used faiss.write_index.
    art = tmp_path / "art"
    embs = _write_artifacts(art)
    faiss.write_index(_build_index(embs, "l2"), str(art / "faiss_index.bin"))
    (art / "faiss_meta.json").write_text(json.dumps({"metric": "l2", "n": 20, "d": 8}))

    index, metric = load_index(str(art))

    assert metric == "l2"
    assert index.ntotal == 20
