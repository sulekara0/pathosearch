"""Thin CLI wrapper (B5): see pathosearch.eval.pooling and pathosearch.eval.run_eval.

Runs the systems on every query, pools their top-``depth`` results plus random control
images, and writes ``pool.jsonl`` and ``overlap.jsonl``. ``splits.json`` must already exist
(it is committed before pooling).

Usage:
    uv run python scripts/build_pool.py --golden eval_data/golden_v0 --depth 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pathosearch.config import load_settings, require_file
from pathosearch.eval.golden import GoldenPaths, load_corpus, load_queries
from pathosearch.eval.pooling import (
    DEFAULT_DEPTH,
    DEFAULT_N_RANDOM,
    DEFAULT_OVERLAP_FRAC,
    DEFAULT_SEED,
    build_pool,
    overlap_subset,
    pool_stats,
    save_pool,
)
from pathosearch.eval.run_eval import build_runners, golden_caption_map, parse_system


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--golden", type=Path, required=True)
    p.add_argument("--systems", default="clip,quilt,clip+rerank,quilt+rerank")
    p.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    p.add_argument("--n_random", type=int, default=DEFAULT_N_RANDOM)
    p.add_argument("--overlap_frac", type=float, default=DEFAULT_OVERLAP_FRAC)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--artifacts-dir", type=Path, default=None)
    args = p.parse_args()

    s = load_settings()
    golden = GoldenPaths(args.golden)
    require_file(golden.splits, "Run make_splits.py and commit splits.json before pooling")
    csv_path = require_file(args.csv or s.csv_path, "Set PATHOSEARCH_CSV_PATH or pass --csv")
    art_root = args.artifacts_dir or (s.artifacts_dir / golden.root.name)

    corpus = load_corpus(golden.corpus)
    queries = load_queries(golden.queries)
    specs = [parse_system(n) for n in args.systems.split(",")]
    runners = build_runners(specs, args.depth, art_root, golden_caption_map(corpus, csv_path))
    runs = {sp.name: {q.qid: runners[sp.name](q.text) for q in queries} for sp in specs}

    pool = build_pool(runs, [e.image for e in corpus], args.depth, args.n_random, args.seed)
    overlap = overlap_subset(pool, args.overlap_frac, args.seed)
    save_pool(golden.pool, pool)
    save_pool(golden.overlap, overlap)
    print(json.dumps({"pool": pool_stats(pool), "overlap_pairs": len(overlap)}, indent=2))


if __name__ == "__main__":
    main()
