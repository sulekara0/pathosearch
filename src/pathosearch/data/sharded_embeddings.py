"""Resumable image embedding in fixed shards (R4: train candidates without embeddings).

Names are sorted and cut into shards of ``shard_size``. Each finished shard is written
atomically as ``part_XXXXX.npy`` + ``part_XXXXX.json`` (temp file, then rename), so an
interrupted run restarts at the first missing shard. ``merge_shards`` checks that every
shard is present and writes the usual ``embeddings.npy`` / ``ids.json`` / ``model_info.json``.

Images that cannot be opened are listed per shard and in ``model_info.json``; they are
never replaced by another image. ``benchmark`` encodes the first N names without writing
shards, to estimate the total time before a long run.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pathosearch.pipeline import ImageEncoder

SHARD_SIZE = 1000
BATCH_SIZE = 64

Log = Callable[[str], None]


def make_shards(names: Sequence[str], shard_size: int = SHARD_SIZE) -> list[list[str]]:
    """Sorted, de-duplicated names cut into consecutive shards."""
    if shard_size < 1:
        raise ValueError("shard_size must be >= 1")
    ordered = sorted(set(names))
    return [ordered[i : i + shard_size] for i in range(0, len(ordered), shard_size)]


def shard_paths(out_dir: Path, i: int) -> tuple[Path, Path]:
    return out_dir / f"part_{i:05d}.npy", out_dir / f"part_{i:05d}.json"


def _save_npy_atomic(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        np.save(fh, arr)
    os.replace(tmp, path)


def _save_json_atomic(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj), encoding="utf-8")
    os.replace(tmp, path)


def encode_names(
    names: Sequence[str],
    paths: Mapping[str, Path],
    encoder: ImageEncoder,
    batch_size: int = BATCH_SIZE,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Encode images in order. Returns (embeddings, encoded ids, names that failed to open)."""
    embs: list[np.ndarray] = []
    ids: list[str] = []
    failed: list[str] = []
    batch: list[Image.Image] = []
    batch_ids: list[str] = []

    def flush() -> None:
        if batch:
            embs.append(np.asarray(encoder.encode_images(batch), dtype=np.float32))
            ids.extend(batch_ids)
            batch.clear()
            batch_ids.clear()

    for name in names:
        try:
            with Image.open(paths[name]) as im:
                batch.append(im.convert("RGB"))
        except (OSError, KeyError):
            failed.append(name)
            continue
        batch_ids.append(name)
        if len(batch) == batch_size:
            flush()
    flush()
    dim = embs[0].shape[1] if embs else 0
    return (np.vstack(embs) if embs else np.zeros((0, dim), np.float32)), ids, failed


def run_shards(
    shards: Sequence[Sequence[str]],
    paths: Mapping[str, Path],
    encoder: ImageEncoder,
    out_dir: Path,
    batch_size: int = BATCH_SIZE,
    log: Log = print,
) -> dict[str, int]:
    """Encode every shard that is not on disk yet. Returns counts of done/skipped shards."""
    out_dir.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    for i, shard in enumerate(shards):
        npy, meta = shard_paths(out_dir, i)
        if npy.is_file() and meta.is_file():
            skipped += 1
            continue
        t0 = time.perf_counter()
        emb, ids, failed = encode_names(shard, paths, encoder, batch_size)
        seconds = time.perf_counter() - t0
        # .npy first: a shard counts as done only when its .json exists too
        _save_npy_atomic(npy, emb)
        record = {"names": list(shard), "ids": ids, "failed": failed, "seconds": seconds}
        _save_json_atomic(meta, record)
        done += 1
        log(
            f"shard {i + 1}/{len(shards)}: {len(ids)} images, {len(failed)} failed, "
            f"{len(shard) / max(seconds, 1e-9):.1f} img/s"
        )
    return {"shards": len(shards), "written": done, "skipped": skipped}


def merge_shards(
    shards: Sequence[Sequence[str]], out_dir: Path, info: Mapping[str, Any]
) -> dict[str, Any]:
    """Concatenate all shards; raise if one is missing or was cut from a different list."""
    embs: list[np.ndarray] = []
    ids: list[str] = []
    failed: list[str] = []
    seconds = 0.0
    for i, shard in enumerate(shards):
        npy, meta = shard_paths(out_dir, i)
        if not (npy.is_file() and meta.is_file()):
            raise FileNotFoundError(f"shard {i} missing in {out_dir}; run the shards first")
        record = json.loads(meta.read_text(encoding="utf-8"))
        if record["names"] != list(shard):
            raise ValueError(f"shard {i} was built from a different name list")
        emb = np.load(npy)
        if len(emb) != len(record["ids"]):
            raise ValueError(f"shard {i}: {len(emb)} vectors for {len(record['ids'])} ids")
        embs.append(emb)
        ids.extend(record["ids"])
        failed.extend(record["failed"])
        seconds += float(record["seconds"])
    nonempty = [e for e in embs if len(e)]
    emb_all = np.vstack(nonempty).astype(np.float32) if nonempty else np.zeros((0, 0), np.float32)
    np.save(out_dir / "embeddings.npy", emb_all)
    (out_dir / "ids.json").write_text(json.dumps(ids, indent=2), encoding="utf-8")
    model_info = {
        **info,
        "n_images": len(ids),
        "n_failed": len(failed),
        "failed": failed,
        "n_shards": len(shards),
        "encode_seconds": round(seconds, 2),
        "creation_time": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "model_info.json").write_text(json.dumps(model_info, indent=2), encoding="utf-8")
    return model_info


def benchmark(
    names: Sequence[str],
    paths: Mapping[str, Path],
    encoder: ImageEncoder,
    n: int,
    total: int,
    batch_size: int = BATCH_SIZE,
) -> dict[str, Any]:
    """Encode the first ``n`` sorted names; estimate the time for ``total`` images."""
    sample = sorted(set(names))[:n]
    t0 = time.perf_counter()
    _, ids, failed = encode_names(sample, paths, encoder, batch_size)
    seconds = time.perf_counter() - t0
    rate = len(sample) / seconds if seconds > 0 else float("inf")
    return {
        "sample": len(sample),
        "encoded": len(ids),
        "failed": len(failed),
        "seconds": round(seconds, 2),
        "images_per_second": round(rate, 2),
        "total_images": total,
        "estimated_total_minutes": round(total / rate / 60, 1),
    }
