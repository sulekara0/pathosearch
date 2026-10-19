"""Thin CLI wrapper: see pathosearch.eval.fake_golden.

Usage (from the repo root; output is under the git-ignored artifacts/ directory):
    uv run python scripts/make_fake_golden.py --out artifacts/fake_golden
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pathosearch.eval.fake_golden import make_fake_golden


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("artifacts/fake_golden"))
    p.add_argument("--n_images", type=int, default=60)
    p.add_argument("--n_queries", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    fake = make_fake_golden(
        args.out, n_images=args.n_images, n_queries=args.n_queries, seed=args.seed
    )
    print(f"wrote fake golden set to {args.out}\nrun the labeling tool with:")
    print(fake.run_command())


if __name__ == "__main__":
    main()
