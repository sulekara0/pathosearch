"""Thin CLI wrapper: see pathosearch.eval.val_baseline.

Usage (from the repo root; CPU, fp32):
    uv run python scripts/eval_val_baseline.py --manifest eval_data/r4_train_v0 \
        --images-dir artifacts/r4_bundle/r4_train_v0_val/images \
        --captions artifacts/r4_bundle/r4_train_v0_val/captions.jsonl

Pass the Colab bundle's images and captions.jsonl so the inputs match the training run's
validation byte for byte. Writes ``reports/r4_val_baseline.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from pathosearch.config import load_settings, require_file
from pathosearch.eval.labeling import build_image_index
from pathosearch.eval.val_baseline import build_base_clip, evaluate_base
from pathosearch.keep_awake import keep_awake
from pathosearch.train.data import ClipCollator, load_captions_jsonl
from pathosearch.train.manifest import load_image_meta
from pathosearch.train.runner import (
    file_sha256,
    git_state,
    load_manifest,
    run_config,
    select_run_names,
    versions,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--images-dir", type=Path, default=None, help="default: PATHOSEARCH_IMAGES_DIR")
    p.add_argument("--captions", type=Path, default=None, help="captions.jsonl (default: CSV)")
    p.add_argument("--out", type=Path, default=Path("reports/r4_val_baseline.json"))
    args = p.parse_args()

    cfg = run_config("10k", 0)  # only base_model, caption_seed and batch_size are used
    s = load_settings()
    m = load_manifest(args.manifest)
    _, val_names = select_run_names(m, cfg)
    images_dir = args.images_dir or s.images_dir
    paths = build_image_index(images_dir)
    if args.captions:
        captions = load_captions_jsonl(args.captions)
    else:
        csv = require_file(s.csv_path, "Set PATHOSEARCH_CSV_PATH or pass --captions")
        meta = load_image_meta(csv, val_names)
        captions = {n: meta[n].captions for n in meta}

    from transformers import CLIPProcessor

    collate = ClipCollator(CLIPProcessor.from_pretrained(cfg.base_model))
    model = build_base_clip(cfg.base_model)
    print(f"base CLIP: {len(val_names)} val images on cpu -> {args.out}")
    t0 = time.perf_counter()
    with keep_awake():
        metrics = evaluate_base(model, cfg, val_names, captions, paths, collate)
    report = {
        "model": cfg.base_model,
        "lora": None,
        "device": "cpu",
        "amp": "off",
        "n_val": len(val_names),
        "caption_seed": cfg.caption_seed,
        "batch_size": cfg.batch_size,
        "logit_scale": float(model.temperature().detach()),
        **metrics,
        "seconds": round(time.perf_counter() - t0, 2),
        "provenance": {
            **git_state(),
            "command": " ".join(["uv run python scripts/eval_val_baseline.py", *sys.argv[1:]]),
            "manifest": args.manifest.as_posix(),
            "val_manifest_sha256": file_sha256(args.manifest / "val_manifest.jsonl"),
            "captions": args.captions.as_posix() if args.captions else "csv",
            "captions_sha256": file_sha256(args.captions) if args.captions else None,
            "images_dir": images_dir.as_posix(),
            "versions": versions(),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
