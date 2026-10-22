"""Glue for ``scripts/train_lora.py``: manifest loading, run names, configs, provenance.

The pre-registered settings (ADR-0004) are the :class:`TrainConfig` defaults; a real run
can change only the scale, the seed and the DataLoader worker count. The smoke config
exists to exercise the whole path on a CPU in a minute and is never reported.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from pathosearch.eval.golden import read_jsonl
from pathosearch.train.config import TrainConfig
from pathosearch.train.data import select_scale

SMOKE_NAME = "smoke"


@dataclass(frozen=True)
class Manifest:
    root: Path
    train_rows: list[dict[str, Any]]
    val_names: list[str]
    scales: dict[str, int]


def load_manifest(root: Path) -> Manifest:
    meta = json.loads((root / "train_manifest_meta.json").read_text(encoding="utf-8"))
    return Manifest(
        root,
        list(read_jsonl(root / "train_manifest.jsonl")),
        [r["image"] for r in read_jsonl(root / "val_manifest.jsonl")],
        {k: int(v) for k, v in meta["scales"].items()},
    )


def run_name(scale: str, seed: int) -> str:
    return f"lora-{scale}-s{seed}"


def run_config(scale: str, seed: int, num_workers: int = 2) -> TrainConfig:
    return TrainConfig(name=run_name(scale, seed), scale=scale, seed=seed, num_workers=num_workers)


def smoke_config() -> TrainConfig:
    return TrainConfig(
        name=SMOKE_NAME,
        scale="10k",
        epochs=1,
        batch_size=8,
        num_workers=0,
        max_steps=2,
        max_train_images=64,
        max_val_images=32,
    )


def select_run_names(m: Manifest, cfg: TrainConfig) -> tuple[list[str], list[str]]:
    if cfg.scale not in m.scales:
        raise ValueError(f"unknown scale {cfg.scale!r}; manifest has {sorted(m.scales)}")
    train = select_scale(m.train_rows, m.scales[cfg.scale])
    val = list(m.val_names)
    if cfg.max_train_images is not None:
        train = train[: cfg.max_train_images]
    if cfg.max_val_images is not None:
        val = val[: cfg.max_val_images]
    return train, val


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state() -> dict[str, Any]:
    """Full HEAD SHA and whether tracked files differ from it (untracked files ignored)."""

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout

    try:
        return {
            "git_sha": git("rev-parse", "HEAD").strip(),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no").strip()),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"git_sha": None, "dirty": None}


def versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for n in ("torch", "transformers", "peft", "numpy", "pillow", "pathosearch"):
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = "not installed"
    return out
