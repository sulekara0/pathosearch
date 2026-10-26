"""Thin CLI wrapper: see pathosearch.eval.r4_coverage.

Ranked lists and judged@k of R4 adapter systems; no retrieval metric is computed.

Usage:
    uv run python scripts/r4_judged_at_k.py --golden eval_data/golden_v0 \
        --adapter lora-10k-s0=artifacts/r4/runs/lora-10k-s0/adapter \
        --adapter lora-30k-s0=artifacts/r4/runs/lora-30k-s0/adapter
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pathosearch.config import load_settings
from pathosearch.eval.golden import GoldenPaths
from pathosearch.eval.r4_coverage import adapter_dirs, coverage_report, split_runs
from pathosearch.eval.run_eval import (
    adapter_summary,
    build_runners,
    command_line,
    git_sha,
    parse_system,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--golden", type=Path, required=True)
    p.add_argument("--adapter", action="append", required=True, metavar="NAME=DIR")
    p.add_argument("--splits", default="dev,test")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--artifacts-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("reports/r4_judged_at_10.json"))
    args = p.parse_args()

    golden = GoldenPaths(args.golden)
    art_root = args.artifacts_dir or (load_settings().artifacts_dir / golden.root.name)
    dirs = adapter_dirs(args.adapter)
    specs = [parse_system(n, extra_encoders=set(dirs)) for n in dirs]
    runners = build_runners(specs, args.k, art_root, {}, dirs)

    runs, judged = split_runs(runners, golden, args.splits.split(","))
    report = {
        "command": command_line(),
        "git_sha": git_sha(),
        "adapters": {n: adapter_summary(d) for n, d in dirs.items()},
        **coverage_report(runs, judged, args.k),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for split, block in report["splits"].items():
        print(f"[{split}] unjudged union: {block['unjudged_union']}")
        for system, s in block["systems"].items():
            print(f"  {system}: judged@{args.k}={s['judged_at_k']} unjudged={s['unjudged_pairs']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
