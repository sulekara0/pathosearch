"""Thin CLI wrapper: see pathosearch.train (ADR-0004 settings).

Usage (from the repo root):
    # local CPU smoke run (64 images, 2 steps); never reported
    uv run python scripts/train_lora.py --manifest eval_data/r4_train_v0 --smoke
    # real run (Colab GPU), bundle from scripts/make_colab_bundle.py
    uv run python scripts/train_lora.py --manifest eval_data/r4_train_v0 --scale 10k --seed 0 \
        --images-dir /content/bundle/images --captions /content/bundle/captions.jsonl \
        --out /content/drive/MyDrive/pathosearch/runs/lora-10k-s0 --resume

``--resume`` continues from ``<out>/checkpoints/last`` when it exists and starts fresh
otherwise, so the same cell can be re-run after a Colab disconnect. A real run refuses a
working tree with uncommitted changes, so ``run.json`` always points at a commit.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch

from pathosearch.config import load_settings, require_file
from pathosearch.eval.labeling import build_image_index
from pathosearch.keep_awake import keep_awake
from pathosearch.train.data import ClipCollator, load_captions_jsonl
from pathosearch.train.loop import train
from pathosearch.train.manifest import load_image_meta
from pathosearch.train.model import build_lora_clip
from pathosearch.train.runner import (
    file_sha256,
    git_state,
    load_manifest,
    run_config,
    select_run_names,
    smoke_config,
    versions,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--scale", choices=["10k", "30k", "full"], default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="tiny CPU run of the whole path")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--images-dir", type=Path, default=None, help="default: PATHOSEARCH_IMAGES_DIR")
    p.add_argument("--captions", type=Path, default=None, help="captions.jsonl (default: CSV)")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--resume", action="store_true", help="continue if a checkpoint exists")
    p.add_argument("--stop-after-epoch", type=int, default=None)
    args = p.parse_args()
    if not args.smoke and args.scale is None:
        p.error("--scale is required unless --smoke")

    cfg = smoke_config() if args.smoke else run_config(args.scale, args.seed, args.num_workers)
    git = git_state()
    if git["dirty"] and not args.smoke:
        sys.exit("working tree has uncommitted changes; commit first so run.json names a SHA")

    s = load_settings()
    m = load_manifest(args.manifest)
    train_names, val_names = select_run_names(m, cfg)
    paths = build_image_index(args.images_dir or s.images_dir)
    if args.captions:
        captions = load_captions_jsonl(args.captions)
    else:
        csv = require_file(s.csv_path, "Set PATHOSEARCH_CSV_PATH or pass --captions")
        meta = load_image_meta(csv, train_names + val_names)
        captions = {n: meta[n].captions for n in meta}

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or (
        s.artifacts_dir
        / "r4"
        / ("smoke" if args.smoke else "runs")
        / (stamp if args.smoke else cfg.name)
    )
    resume = args.resume and (out / "checkpoints" / "last" / "state.pt").is_file()

    from transformers import CLIPProcessor

    collate = ClipCollator(CLIPProcessor.from_pretrained(cfg.base_model))
    provenance = {
        **git,
        "command": " ".join(["uv run python scripts/train_lora.py", *sys.argv[1:]]),
        "smoke": args.smoke,
        "manifest": args.manifest.as_posix(),
        "train_manifest_sha256": file_sha256(args.manifest / "train_manifest.jsonl"),
        "val_manifest_sha256": file_sha256(args.manifest / "val_manifest.jsonl"),
        "captions": args.captions.as_posix() if args.captions else "csv",
        "captions_sha256": file_sha256(args.captions) if args.captions else None,
        "images_dir": (args.images_dir or s.images_dir).as_posix(),
        "versions": versions(),
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{cfg.name}: {len(train_names)} train / {len(val_names)} val on {device} -> {out}")
    with keep_awake():
        rec = train(
            cfg,
            train_names,
            val_names,
            captions,
            paths,
            out,
            build_model=lambda: build_lora_clip(cfg.base_model, cfg.lora_settings()),
            collate=collate,
            device=device,
            provenance=provenance,
            resume=resume,
            stop_after_epoch=args.stop_after_epoch,
        )
    print(
        f"{rec['status']}: step {rec['global_step']}/{rec['total_steps']}, best epoch "
        f"{rec['best_epoch']} (val loss {rec['best_val_loss']}), {rec['seconds']} s -> {out}"
    )


if __name__ == "__main__":
    main()
