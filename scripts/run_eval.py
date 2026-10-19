"""Thin CLI wrapper: see pathosearch.eval.run_eval.

Usage (from the repo root; paths come from .env / PATHOSEARCH_* variables):
    uv run python scripts/run_eval.py --golden eval_data/golden_v0 --split dev \
        --systems clip,quilt,clip+rerank,quilt+rerank --k 10

Golden-corpus embeddings/indexes are read from ``artifacts/<golden dir name>/<encoder>/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pathosearch.config import load_settings, require_dir, require_file
from pathosearch.eval.golden import GoldenPaths, load_corpus
from pathosearch.eval.run_eval import (
    check_split_allowed,
    command_line,
    golden_caption_map,
    make_runner,
    parse_system,
    run_eval,
    write_report,
)
from pathosearch.pipeline import SearchIndex


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
    args = p.parse_args()
    check_split_allowed(args.split, args.i_know_this_is_test)  # fail before loading models

    # heavy imports only after argument/guard checks
    from pathosearch.encoders import get_encoder
    from pathosearch.faiss_search import load_embeddings, load_index

    s = load_settings()
    golden = GoldenPaths(args.golden)
    csv_path = require_file(args.csv or s.csv_path, "Set PATHOSEARCH_CSV_PATH or pass --csv")
    art_root = args.artifacts_dir or (s.artifacts_dir / golden.root.name)
    captions = golden_caption_map(load_corpus(golden.corpus), csv_path)

    specs = [parse_system(n) for n in args.systems.split(",")]
    runners = {}
    for enc_name in sorted({sp.encoder for sp in specs}):
        art_dir = require_dir(art_root / enc_name, "Run B2 (golden embeddings + index) first")
        index, metric = load_index(str(art_dir))
        _, ids = load_embeddings(str(art_dir))
        idx = SearchIndex(index, metric, ids, captions)
        encoder = get_encoder(enc_name)
        for sp in specs:
            if sp.encoder == enc_name:
                runners[sp.name] = make_runner(encoder, idx, sp.rerank, args.k)

    report = run_eval(
        golden,
        args.split,
        [sp.name for sp in specs],
        args.k,
        runners,
        {img: rec["caption"] for img, rec in captions.items()},
        command_line(),
        i_know_this_is_test=args.i_know_this_is_test,
    )
    json_path, md_path = write_report(report, args.out_dir)
    print(f"wrote {json_path}\nwrote {md_path}")


if __name__ == "__main__":
    main()
