"""Thin CLI wrapper: see pathosearch.data.source_groups.

Usage (from the repo root; paths come from .env / PATHOSEARCH_* variables):
    uv run python scripts/analyze_source_groups.py --golden eval_data/golden_v0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pathosearch.config import load_settings, require_dir, require_file
from pathosearch.data.source_groups import source_overlap_report
from pathosearch.eval.corpus_analysis import list_image_names, scan_csv
from pathosearch.eval.golden import GoldenPaths, load_corpus


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--golden", type=Path, required=True)
    p.add_argument(
        "--candidates-ids",
        type=Path,
        default=None,
        help="ids.json of the golden candidates before exclusion "
        "(default: <artifacts>/<golden name>_candidates/clip/ids.json)",
    )
    p.add_argument("--encoder", default="clip", help="artifacts/<encoder>/ids.json = main corpus")
    p.add_argument("--out", type=Path, default=Path("reports/source_groups.json"))
    args = p.parse_args()

    s = load_settings()
    golden = GoldenPaths(args.golden)
    cand_path = args.candidates_ids or (
        s.artifacts_dir / f"{golden.root.name}_candidates" / "clip" / "ids.json"
    )
    ids_path = require_file(s.encoder_dir(args.encoder) / "ids.json")
    disk, _ = list_image_names(require_dir(s.images_dir, "Set PATHOSEARCH_IMAGES_DIR in .env"))
    scan = scan_csv(require_file(s.csv_path, "Set PATHOSEARCH_CSV_PATH in .env"), disk)
    subset: dict[str, str] = {}
    for name, sub in zip(scan.disk_rows["image_path"], scan.disk_rows["subset"], strict=True):
        subset.setdefault(name, str(sub))

    report = source_overlap_report(
        image_mask=scan.image_mask,
        disk=disk,
        subset=subset,
        golden_candidates=json.loads(require_file(cand_path).read_text(encoding="utf-8")),
        golden=[e.image for e in load_corpus(golden.corpus)],
        main_ids=json.loads(ids_path.read_text(encoding="utf-8")),
    )
    report = {
        "command": f"uv run python scripts/analyze_source_groups.py --golden {args.golden.as_posix()}",
        "inputs": {
            "csv": s.csv_path.as_posix(),
            "images_dir": s.images_dir.as_posix(),
            "golden_corpus": golden.corpus.as_posix(),
            "golden_candidates_ids": cand_path.as_posix(),
            "main_ids": ids_path.as_posix(),
        },
        **report,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
