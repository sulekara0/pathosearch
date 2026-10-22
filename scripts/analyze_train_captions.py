"""Thin CLI wrapper: see pathosearch.train.caption_stats.

Usage (from the repo root; the CSV path comes from .env / PATHOSEARCH_CSV_PATH):
    uv run python scripts/analyze_train_captions.py --manifest eval_data/r4_train_v0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pathosearch.config import load_settings, require_file
from pathosearch.eval.golden import read_jsonl
from pathosearch.train.caption_stats import SHORT_CHARS, TOP_N, caption_stats
from pathosearch.train.manifest import load_image_meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--short-chars", type=int, default=SHORT_CHARS)
    p.add_argument("--top-n", type=int, default=TOP_N)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    csv_path = require_file(load_settings().csv_path, "Set PATHOSEARCH_CSV_PATH in .env")
    names = {
        split: [r["image"] for r in read_jsonl(args.manifest / f"{split}_manifest.jsonl")]
        for split in ("train", "val")
    }
    meta = load_image_meta(csv_path, names["train"] + names["val"])
    report = {
        "command": "uv run python scripts/analyze_train_captions.py "
        f"--manifest {args.manifest.as_posix()} --short-chars {args.short_chars} "
        f"--top-n {args.top_n}",
        **{
            split: caption_stats({n: meta[n].captions for n in ns}, args.short_chars, args.top_n)
            for split, ns in names.items()
        },
    }
    out = args.out or Path("reports") / f"train_captions_{args.manifest.name}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
