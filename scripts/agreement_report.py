"""Thin CLI wrapper (B7): see pathosearch.eval.agreement.

Computes Cohen's kappa between annotators A and B (before adjudication) on exactly the
pairs in ``overlap.jsonl`` and writes ``reports/agreement_<golden>.json`` and ``.md``.
With ``--write_qrels`` it also writes ``qrels.tsv`` (A's labels overridden by
``adjudication.jsonl``) once every disagreement is adjudicated.

Usage:
    uv run python scripts/agreement_report.py --golden eval_data/golden_v0 --b claude
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pathosearch.eval.agreement import (
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    agreement_limitations,
    agreement_report,
    final_qrels,
    overlap_agreement,
    render_agreement_markdown,
    unresolved,
)
from pathosearch.eval.golden import (
    GoldenPaths,
    load_judgments,
    load_queries,
    read_jsonl,
    write_trec_qrels,
)
from pathosearch.eval.run_eval import command_line, git_sha

# Facts about who labelled golden_v0; not derivable from the judgment files.
ANNOTATOR_CAVEATS = (
    "Neither annotator is a pathologist. High agreement means the two read the grading "
    "guideline the same way, not that the grades are clinically correct.",
    "Annotator 'claude' is an LLM judging blind (caption_seen=false). Agreement with it "
    "is human-model agreement, not inter-human agreement: a model can share systematic "
    "errors with the human or have its own consistent bias, so it is not a substitute for "
    "a second human annotator.",
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--golden", type=Path, required=True)
    p.add_argument("--a", default="A", help="primary annotator id")
    p.add_argument("--b", default="B", help="second annotator id")
    p.add_argument("--n-boot", type=int, default=BOOTSTRAP_N)
    p.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    p.add_argument("--out-dir", type=Path, default=Path("reports"))
    p.add_argument("--write_qrels", action="store_true")
    args = p.parse_args()

    golden = GoldenPaths(args.golden)
    ja = load_judgments(golden.judgments(args.a))
    jb = load_judgments(golden.judgments(args.b))
    overlap = {(r["qid"], r["image"]) for r in read_jsonl(golden.overlap)}
    queries = load_queries(golden.queries)
    adjudication = load_judgments(golden.adjudication) if golden.adjudication.is_file() else []

    in_overlap = [x for x in ja if x.key in overlap], [x for x in jb if x.key in overlap]
    report = agreement_report(*in_overlap)
    todo = unresolved(report, adjudication)
    detail = overlap_agreement(
        *in_overlap,
        overlap,
        {q.qid: q.category for q in queries},
        {q.qid: q.text for q in queries},
        n_boot=args.n_boot,
        seed=args.seed,
    )
    doc = {
        "golden": golden.root.name,
        "command": command_line(),
        "git_sha": git_sha(),
        "annotators": {"a": args.a, "b": args.b},
        "adjudicated_pairs": len(adjudication),
        "unresolved_disagreements": len(todo),
        "pairs": report.pairs,
        **detail,
        "limitations": agreement_limitations(detail, args.a, args.b, ANNOTATOR_CAVEATS),
    }
    stem = args.out_dir / f"agreement_{golden.root.name}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    json_path, md_path = stem.with_suffix(".json"), stem.with_suffix(".md")
    json_path.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    md_path.write_text(render_agreement_markdown(doc), encoding="utf-8", newline="\n")
    o = detail["ordinal"]
    print(
        f"wrote {json_path} and {md_path}: kappa={o['kappa']['value']:.3f} "
        f"weighted={o['kappa_linear']['value']:.3f} pairs={detail['comparable_pairs']} "
        f"unresolved={len(todo)}"
    )

    if args.write_qrels:
        if todo:
            raise SystemExit(f"{len(todo)} disagreements are not adjudicated; qrels not written")
        qrels = final_qrels(ja, adjudication)
        write_trec_qrels(golden.qrels, qrels)
        print(f"wrote {golden.qrels} ({sum(len(v) for v in qrels.values())} judged pairs)")


if __name__ == "__main__":
    main()
