"""Thin CLI wrapper (B7): see pathosearch.eval.agreement.

Computes Cohen's kappa between annotators A and B (before adjudication) and writes
``reports/agreement_<golden>.json``. With ``--write_qrels`` it also writes ``qrels.tsv``
(A's labels overridden by ``adjudication.jsonl``) once every disagreement is adjudicated.

Usage:
    uv run python scripts/agreement_report.py --golden eval_data/golden_v0 [--write_qrels]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pathosearch.eval.agreement import agreement_report, final_qrels, unresolved
from pathosearch.eval.golden import GoldenPaths, load_judgments, write_trec_qrels
from pathosearch.eval.run_eval import command_line


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--golden", type=Path, required=True)
    p.add_argument("--a", default="A", help="primary annotator id")
    p.add_argument("--b", default="B", help="second annotator id")
    p.add_argument("--out-dir", type=Path, default=Path("reports"))
    p.add_argument("--write_qrels", action="store_true")
    args = p.parse_args()

    golden = GoldenPaths(args.golden)
    ja = load_judgments(golden.judgments(args.a))
    jb = load_judgments(golden.judgments(args.b))
    adjudication = load_judgments(golden.adjudication) if golden.adjudication.is_file() else []

    report = agreement_report(ja, jb)
    todo = unresolved(report, adjudication)
    doc = {
        "golden": golden.root.name,
        "command": command_line(),
        "annotators": {"a": args.a, "b": args.b},
        "adjudicated_pairs": len(adjudication),
        "unresolved_disagreements": len(todo),
        **report.as_dict(),
    }
    out = args.out_dir / f"agreement_{golden.root.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        f"wrote {out}: kappa={report.kappa:.3f} weighted={report.kappa_linear:.3f} "
        f"pairs={report.pairs} unresolved={len(todo)}"
    )

    if args.write_qrels:
        if todo:
            raise SystemExit(f"{len(todo)} disagreements are not adjudicated; qrels not written")
        qrels = final_qrels(ja, adjudication)
        write_trec_qrels(golden.qrels, qrels)
        print(f"wrote {golden.qrels} ({sum(len(v) for v in qrels.values())} judged pairs)")


if __name__ == "__main__":
    main()
