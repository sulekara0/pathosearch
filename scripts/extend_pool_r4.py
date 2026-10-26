"""Thin CLI wrapper: see pathosearch.eval.pooling.extend_pool and pathosearch.eval.r4_coverage.

Appends the adapter systems' unjudged top-``depth`` (query, image) pairs to ``pool.jsonl``
(source ``r4_extension``) so annotator A can judge them blind with ``label_app.py``.
``overlap.jsonl`` is left unchanged (only A labels the extension, ADR-0004 §8).

Usage:
    uv run python scripts/extend_pool_r4.py --golden eval_data/golden_v0 \
        --adapter lora-10k-s0=artifacts/r4/runs/lora-10k-s0/adapter \
        --adapter lora-30k-s0=artifacts/r4/runs/lora-30k-s0/adapter
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from pathosearch.config import load_settings
from pathosearch.eval.golden import GoldenPaths
from pathosearch.eval.pooling import DEFAULT_DEPTH, extend_pool, load_pool, save_pool
from pathosearch.eval.r4_coverage import adapter_dirs, split_runs
from pathosearch.eval.run_eval import build_runners, parse_system


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--golden", type=Path, required=True)
    p.add_argument("--adapter", action="append", required=True, metavar="NAME=DIR")
    p.add_argument("--splits", default="dev,test")
    p.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    p.add_argument("--artifacts-dir", type=Path, default=None)
    args = p.parse_args()

    golden = GoldenPaths(args.golden)
    art_root = args.artifacts_dir or (load_settings().artifacts_dir / golden.root.name)
    dirs = adapter_dirs(args.adapter)
    specs = [parse_system(n, extra_encoders=set(dirs)) for n in dirs]
    runners = build_runners(specs, args.depth, art_root, {}, dirs)

    pool = load_pool(golden.pool)
    runs_by_split, judged_by_split = split_runs(runners, golden, args.splits.split(","))
    added = []
    for split, runs in runs_by_split.items():
        new = extend_pool(pool, runs, judged_by_split[split], args.depth)
        added += new
        print(f"[{split}] added {len(new)} pairs")
    save_pool(golden.pool, [*pool, *added])
    per_source = Counter(it.source for it in [*pool, *added])
    print(json.dumps({"pool_pairs": len(pool) + len(added), "by_source": per_source}, indent=2))


if __name__ == "__main__":
    main()
