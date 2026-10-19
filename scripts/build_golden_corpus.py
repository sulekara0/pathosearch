"""Thin CLI wrapper (B1): see pathosearch.eval.heldout.

Selects the held-out corpus with a named rule and writes ``corpus_manifest.jsonl`` and
``manifest_meta.json``. Paths come from .env / PATHOSEARCH_* variables.

Optional ADR-0003 steps (default off; need candidate embeddings from B2 for the *unfiltered* rule
output, in ``--candidates-dir``):
    --exclude-near-dup <tau>  drop images with cos >= tau to the main corpus in ANY encoder
    --dedup-within <tau>      keep one representative per cluster (cos >= tau in ANY encoder)

Usage:
    uv run python scripts/build_golden_corpus.py --rule val_only_on_disk --out eval_data/golden_v0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pathosearch.config import load_settings, require_dir, require_file
from pathosearch.eval.corpus_analysis import list_image_names, scan_csv
from pathosearch.eval.dedup import dedup_within, exclude_near_dup
from pathosearch.eval.golden import GoldenPaths, save_corpus
from pathosearch.eval.heldout import RULES, compare_with_analysis, manifest_meta, select_heldout
from pathosearch.faiss_search import load_embeddings


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
    p.add_argument("--exclude-near-dup", type=float, default=None, metavar="TAU")
    p.add_argument("--dedup-within", type=float, default=None, metavar="TAU")
    p.add_argument(
        "--candidates-dir",
        type=Path,
        default=None,
        help="artifacts/<name>_candidates/<encoder>/ with embeddings of the unfiltered rule output",
    )
    p.add_argument("--encoders", default="clip,quilt", help="encoders used by the ADR-0003 steps")
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
    candidates = len(entries)
    excluded: list[dict] = []
    dropped: list[dict] = []
    if args.exclude_near_dup is not None or args.dedup_within is not None:
        cand_dir = args.candidates_dir or (s.artifacts_dir / f"{golden.root.name}_candidates")
        encs = args.encoders.split(",")
        g_emb, m_emb = {}, {}
        for enc in encs:
            g, g_ids = load_embeddings(str(require_dir(cand_dir / enc, "Run B2 on the candidates")))
            if sorted(g_ids) != sorted(e.image for e in entries):
                raise SystemExit(f"{enc}: candidate embeddings do not match the rule output")
            order = {n: i for i, n in enumerate(g_ids)}
            g_emb[enc] = g[[order[e.image] for e in entries]]
            m_emb[enc], main_ids = load_embeddings(str(require_dir(s.encoder_dir(enc))))
        names = [e.image for e in entries]
        if args.exclude_near_dup is not None:
            excluded = exclude_near_dup(g_emb, names, m_emb, main_ids, args.exclude_near_dup)
            gone = {r["image"] for r in excluded}
            keep = [i for i, n in enumerate(names) if n not in gone]
            entries = [entries[i] for i in keep]
            g_emb = {k: v[keep] for k, v in g_emb.items()}
            names = [names[i] for i in keep]
        if args.dedup_within is not None:
            kept, dropped = dedup_within(g_emb, names, args.dedup_within)
            keep_set = set(kept)
            entries = [e for e in entries if e.image in keep_set]
    save_corpus(golden.corpus, entries)
    meta = manifest_meta(args.rule, entries, {"csv": csv_path, "ids": ids_path}, len(disk))
    cmd = f"uv run python scripts/build_golden_corpus.py --rule {args.rule} --out {args.out}"
    if args.exclude_near_dup is not None:
        cmd += f" --exclude-near-dup {args.exclude_near_dup}"
    if args.dedup_within is not None:
        cmd += f" --dedup-within {args.dedup_within}"
    meta["command"] = cmd
    if excluded or dropped or args.exclude_near_dup is not None or args.dedup_within is not None:
        meta["candidates_before_filters"] = candidates
        meta["near_dup_exclusion"] = {
            "tau": args.exclude_near_dup,
            "rule": "cos >= tau to main corpus in any of: " + args.encoders,
            "n_excluded": len(excluded),
            "list": "excluded_near_dup.jsonl",
        }
        meta["dedup_within"] = {
            "tau": args.dedup_within,
            "rule": "one representative per cluster, smallest file name",
            "n_dropped": len(dropped),
            "list": "dedup_within_dropped.jsonl",
        }
        for fname, rows in (
            ("excluded_near_dup.jsonl", excluded),
            ("dedup_within_dropped.jsonl", dropped),
        ):
            (golden.root / fname).write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                encoding="utf-8",
                newline="\n",
            )
    (golden.root / "manifest_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {golden.corpus} ({len(entries)} images) and manifest_meta.json")


if __name__ == "__main__":
    main()
