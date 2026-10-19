"""Thin CLI wrapper (B3): see pathosearch.eval.heldout.

Runs the leakage checks (a)-(d) for a golden corpus and writes
``reports/leakage_<golden>.json``. Exits non-zero if any check fails.
Golden embeddings are read from ``artifacts/<golden>/<encoder>/``, the main corpus from
``artifacts/<encoder>/``.

Usage:
    uv run python scripts/leakage_report.py --golden eval_data/golden_v0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pathosearch.config import load_settings, require_dir, require_file
from pathosearch.eval.corpus_analysis import scan_csv
from pathosearch.eval.golden import GoldenPaths, load_corpus
from pathosearch.eval.heldout import NEAR_DUP_TAU, build_leakage_report
from pathosearch.eval.run_eval import command_line
from pathosearch.faiss_search import load_embeddings


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--golden", type=Path, required=True)
    p.add_argument("--encoders", default="clip,quilt")
    p.add_argument("--tau", type=float, default=NEAR_DUP_TAU)
    p.add_argument("--artifacts-dir", type=Path, default=None, help="golden artifacts root")
    p.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = p.parse_args()

    s = load_settings()
    golden = GoldenPaths(args.golden)
    golden_art = args.artifacts_dir or (s.artifacts_dir / golden.root.name)
    golden_ids = [e.image for e in load_corpus(golden.corpus)]
    csv_path = require_file(s.csv_path, "Set PATHOSEARCH_CSV_PATH in .env")
    scan = scan_csv(csv_path, set(golden_ids))

    golden_emb, main_emb = {}, {}
    for enc in args.encoders.split(","):
        golden_emb[enc] = load_embeddings(str(require_dir(golden_art / enc, "Run B2 first")))
        main_emb[enc] = load_embeddings(str(require_dir(s.encoder_dir(enc))))
    main_ids = next(iter(main_emb.values()))[1]

    report = build_leakage_report(
        golden_ids, main_ids, scan.image_mask, golden_emb, main_emb, args.tau
    )
    report["golden"] = golden.root.name
    report["command"] = command_line()
    out = args.out_dir / f"leakage_{golden.root.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {out}: passed={report['passed']} {report['violation_counts']}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
