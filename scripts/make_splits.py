"""Thin CLI wrapper: see pathosearch.eval.splits.

Writes ``splits.json`` (stratified dev/test split of query ids) and refuses to overwrite
an existing one: the split is committed before pooling and must not change afterwards.

Usage:
    uv run python scripts/make_splits.py --golden eval_data/golden_v0 \
        [--dev_frac 0.5] [--force_dev q001,q002]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pathosearch.eval.golden import GoldenPaths, load_queries
from pathosearch.eval.splits import (
    DEFAULT_SEED,
    assert_no_split_leakage,
    default_dev_frac,
    save_splits,
    split_queries,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--golden", type=Path, required=True)
    p.add_argument("--dev_frac", type=float, default=None)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--force_dev", default="", help="comma-separated qids kept in dev (pilot)")
    p.add_argument("--overwrite", action="store_true", help="replace an existing splits.json")
    args = p.parse_args()

    golden = GoldenPaths(args.golden)
    if golden.splits.exists() and not args.overwrite:
        raise SystemExit(
            f"{golden.splits} exists; the split must not change (--overwrite to force)"
        )
    queries = load_queries(golden.queries)
    dev_frac = args.dev_frac if args.dev_frac is not None else default_dev_frac(len(queries))
    forced = [q for q in args.force_dev.split(",") if q]
    splits = split_queries(queries, dev_frac, args.seed, forced)
    report = assert_no_split_leakage(queries, splits)  # raises on errors
    save_splits(golden.splits, splits, args.seed, dev_frac)
    print(f"wrote {golden.splits}: dev={len(splits['dev'])} test={len(splits['test'])}")
    for a, b, score, crosses in report.near_duplicates:
        where = "ACROSS dev/test" if crosses else "same split"
        print(f"warning: near-duplicate queries {a} / {b} (token Jaccard {score}, {where})")


if __name__ == "__main__":
    main()
