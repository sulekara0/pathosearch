"""Thin CLI wrapper: see pathosearch.eval.run_eval.

Usage (from the repo root; paths come from .env / PATHOSEARCH_* variables):
    uv run python scripts/run_eval.py --golden eval_data/golden_v0 --split dev \
        --systems clip,quilt,clip+rerank,quilt+rerank --k 10

R4 adapters (golden index in ``artifacts/<golden>/<NAME>/``):
    uv run python scripts/run_eval.py --golden eval_data/golden_v0 --split dev \
        --systems clip,quilt,lora-full-s0 --adapter lora-full-s0=artifacts/r4/runs/full_s0/adapter \
        --compare lora-full-s0:clip --compare lora-full-s0:quilt

Golden-corpus embeddings/indexes are read from ``artifacts/<golden dir name>/<encoder>/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pathosearch.config import load_settings, require_file
from pathosearch.eval.golden import GoldenPaths, load_corpus
from pathosearch.eval.run_eval import (
    adapter_summary,
    build_runners,
    check_split_allowed,
    command_line,
    golden_caption_map,
    parse_system,
    run_eval,
    write_report,
)


def parse_pair(text: str, sep: str, usage: str) -> tuple[str, str]:
    a, found, b = text.partition(sep)
    if not found or not a or not b:
        raise SystemExit(f"expected {usage}, got {text!r}")
    return a, b


def parse_pairs(items: list[str], sep: str, usage: str) -> dict[str, str]:
    return dict(parse_pair(t, sep, usage) for t in items)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--golden", type=Path, required=True)
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--systems", default="clip,quilt,clip+rerank,quilt+rerank")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--csv", type=Path, default=None, help="Quilt lookup CSV (default: settings)")
    p.add_argument("--artifacts-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("reports"))
    p.add_argument("--i-know-this-is-test", action="store_true")
    p.add_argument(
        "--adapter",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help="register an R4 LoRA adapter as system NAME (repeatable)",
    )
    p.add_argument(
        "--compare",
        action="append",
        default=None,
        metavar="A:B",
        help="paired difference A - B (repeatable; default: rerank vs base, others vs first)",
    )
    args = p.parse_args()
    check_split_allowed(args.split, args.i_know_this_is_test)  # fail before loading models
    adapters = parse_pairs(args.adapter, "=", "--adapter NAME=DIR")
    comparisons = (
        [parse_pair(c, ":", "--compare A:B") for c in args.compare] if args.compare else None
    )

    s = load_settings()
    golden = GoldenPaths(args.golden)
    csv_path = require_file(args.csv or s.csv_path, "Set PATHOSEARCH_CSV_PATH or pass --csv")
    art_root = args.artifacts_dir or (s.artifacts_dir / golden.root.name)
    captions = golden_caption_map(load_corpus(golden.corpus), csv_path)

    adapter_dirs = {n: Path(d) for n, d in adapters.items()}
    adapter_info = {n: adapter_summary(d) for n, d in adapter_dirs.items()}
    specs = [parse_system(n, extra_encoders=set(adapters)) for n in args.systems.split(",")]
    runners = build_runners(specs, args.k, art_root, captions, adapter_dirs)

    report = run_eval(
        golden,
        args.split,
        [sp.name for sp in specs],
        args.k,
        runners,
        {img: rec["caption"] for img, rec in captions.items()},
        command_line(),
        i_know_this_is_test=args.i_know_this_is_test,
        comparisons=comparisons,
        adapters=adapter_info,
    )
    json_path, md_path = write_report(report, args.out_dir)
    print(f"wrote {json_path}\nwrote {md_path}")


if __name__ == "__main__":
    main()
