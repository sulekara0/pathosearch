"""Thin CLI wrapper (B1): see pathosearch.eval.heldout.

Selects the held-out corpus with a named rule and writes ``corpus_manifest.jsonl`` and
``manifest_meta.json``. Paths come from .env / PATHOSEARCH_* variables.

Usage:
    uv run python scripts/build_golden_corpus.py --rule val_only_on_disk --out eval_data/golden_v0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pathosearch.config import load_settings, require_dir, require_file
from pathosearch.eval.corpus_analysis import list_image_names, scan_csv
from pathosearch.eval.golden import GoldenPaths, save_corpus
from pathosearch.eval.heldout import RULES, compare_with_analysis, manifest_meta, select_heldout


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rule", choices=sorted(RULES), default="val_only_on_disk")
    p.add_argument("--out", type=Path, required=True, help="golden dir, e.g. eval_data/golden_v0")
    p.add_argument("--encoder", default="clip", help="artifacts/<encoder>/ids.json = main corpus")
    p.add_argument(
        "--check_against",
        type=Path,
        default=Path("reports/corpus_analysis.json"),
        help="analysis report whose counts the manifest must match",
    )
    p.add_argument("--no_check", action="store_true", help="skip the --check_against comparison")
    args = p.parse_args()

    s = load_settings()
    csv_path = require_file(s.csv_path, "Set PATHOSEARCH_CSV_PATH in .env")
    ids_path = require_file(s.encoder_dir(args.encoder) / "ids.json")
    disk, _ = list_image_names(require_dir(s.images_dir, "Set PATHOSEARCH_IMAGES_DIR in .env"))
    scan = scan_csv(csv_path, disk)
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    entries = select_heldout(scan, disk, ids, args.rule)

    if not args.no_check:
        analysis = json.loads(require_file(args.check_against).read_text(encoding="utf-8"))
        problems = compare_with_analysis(entries, analysis)
        if problems:
            raise SystemExit("manifest differs from the corpus analysis:\n" + "\n".join(problems))
        print(f"counts match {args.check_against}")

    golden = GoldenPaths(args.out)
    save_corpus(golden.corpus, entries)
    meta = manifest_meta(args.rule, entries, {"csv": csv_path, "ids": ids_path}, len(disk))
    meta["command"] = (
        f"uv run python scripts/build_golden_corpus.py --rule {args.rule} --out {args.out}"
    )
    (golden.root / "manifest_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {golden.corpus} ({len(entries)} images) and manifest_meta.json")


if __name__ == "__main__":
    main()
