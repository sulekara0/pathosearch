"""Thin CLI wrapper: see pathosearch.eval.corpus_analysis.

Usage (from the repo root; paths come from .env / PATHOSEARCH_* variables):
    uv run python scripts/analyze_corpus.py --encoder clip --out reports/corpus_analysis.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pathosearch.config import load_settings, require_dir, require_file
from pathosearch.eval.corpus_analysis import MIN_PER_CATEGORY, run_analysis


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder", default="clip", help="artifacts/<encoder>/ids.json to use")
    p.add_argument("--min_per_category", type=int, default=MIN_PER_CATEGORY)
    p.add_argument("--out", type=Path, default=Path("reports/corpus_analysis.json"))
    args = p.parse_args()

    s = load_settings()
    result = run_analysis(
        csv_path=require_file(s.csv_path, "Set PATHOSEARCH_CSV_PATH in .env"),
        captions_path=require_file(s.captions_path, "Set PATHOSEARCH_CAPTIONS_PATH in .env"),
        images_dir=require_dir(s.images_dir, "Set PATHOSEARCH_IMAGES_DIR in .env"),
        ids_path=require_file(s.encoder_dir(args.encoder) / "ids.json"),
        min_per_category=args.min_per_category,
    )
    result["command"] = (
        f"uv run python scripts/analyze_corpus.py --encoder {args.encoder} "
        f"--min_per_category {args.min_per_category}"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
