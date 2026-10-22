"""Thin CLI wrapper: see pathosearch.data.sharded_embeddings.

Usage (from the repo root; paths come from .env / PATHOSEARCH_* variables):
    # 1) names of on-disk train-only images that have no main-corpus embedding
    uv run python scripts/embed_sharded.py list --out artifacts/r4_train_candidates/missing.txt
    # 2) time 500 images, then run (resumable) and merge, per encoder
    uv run python scripts/embed_sharded.py bench --names <txt> --encoder clip --n 500
    uv run python scripts/embed_sharded.py run   --names <txt> --encoder clip
    uv run python scripts/embed_sharded.py merge --names <txt> --encoder clip
    # 3) one-line progress
    uv run python scripts/embed_sharded.py progress --names <txt>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from pathosearch.config import load_settings, require_dir, require_file
from pathosearch.data.sharded_embeddings import (
    BATCH_SIZE,
    SHARD_SIZE,
    benchmark,
    make_shards,
    merge_shards,
    progress,
    run_shards,
)
from pathosearch.eval.corpus_analysis import list_image_names, scan_csv, split_class
from pathosearch.eval.labeling import build_image_index
from pathosearch.eval.run_eval import git_sha, package_versions
from pathosearch.keep_awake import keep_awake


def read_names(path: Path) -> list[str]:
    text = require_file(path, "Run the 'list' step first").read_text(encoding="utf-8")
    return [line for line in text.splitlines() if line.strip()]


def cmd_list(args: argparse.Namespace) -> None:
    s = load_settings()
    disk, _ = list_image_names(require_dir(s.images_dir, "Set PATHOSEARCH_IMAGES_DIR in .env"))
    scan = scan_csv(require_file(s.csv_path, "Set PATHOSEARCH_CSV_PATH in .env"), set())
    train = {
        n for n in disk if n in scan.image_mask and split_class(scan.image_mask[n]) == "train_only"
    }
    ids_path = require_file(s.encoder_dir(args.main_encoder) / "ids.json")
    main = set(json.loads(ids_path.read_text(encoding="utf-8")))
    missing = sorted(train - main)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(missing) + "\n", encoding="utf-8")
    print(
        f"disk {len(disk)} | train-only on disk {len(train)} | with main embedding "
        f"{len(train & main)} | missing {len(missing)} -> {args.out}"
    )


def cmd_shards(args: argparse.Namespace) -> None:
    from pathosearch.encoders import get_encoder

    s = load_settings()
    names = read_names(args.names)
    out_dir = args.out or (args.names.parent / args.encoder)
    shards = make_shards(names, args.shard_size)
    if args.cmd == "merge":
        info = {
            "encoder": args.encoder,
            "names_file": args.names.as_posix(),
            "names_sha256": hashlib.sha256(args.names.read_bytes()).hexdigest(),
            "shard_size": args.shard_size,
            "batch_size": args.batch_size,
            "git_sha": git_sha(),
            "packages": package_versions(),
            "command": " ".join(["uv run python scripts/embed_sharded.py", *sys.argv[1:]]),
        }
        out = merge_shards(shards, out_dir, info)
        print(f"merged {out['n_images']} images ({out['n_failed']} failed) -> {out_dir}")
        return

    paths = build_image_index(s.images_dir)
    encoder = get_encoder(args.encoder)
    if args.cmd == "bench":
        result = {"encoder": encoder.info(), **benchmark(names, paths, encoder, args.n, len(names))}
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "timing.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return
    with keep_awake() as awake:
        print(f"sleep block: {'on' if awake else 'unavailable'}", flush=True)
        print(run_shards(shards, paths, encoder, out_dir, args.batch_size))


def cmd_progress(args: argparse.Namespace) -> None:
    shards = make_shards(read_names(args.names), args.shard_size)
    parts = []
    for enc in args.encoders.split(","):
        done, total = progress(shards, args.names.parent / enc)
        merged = (args.names.parent / enc / "model_info.json").is_file()
        parts.append(f"{enc} {done}/{total}{' merged' if merged else ''}")
    print(" | ".join(parts))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("list", help="write names of train-only disk images missing embeddings")
    pl.add_argument("--out", type=Path, required=True)
    pl.add_argument("--main-encoder", default="clip", help="artifacts/<encoder>/ids.json")
    pl.set_defaults(func=cmd_list)
    pp = sub.add_parser("progress", help="one line: finished shards per encoder")
    pp.add_argument("--names", type=Path, required=True)
    pp.add_argument("--encoders", default="clip,quilt")
    pp.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    pp.set_defaults(func=cmd_progress)
    for name in ("bench", "run", "merge"):
        q = sub.add_parser(name)
        q.add_argument("--names", type=Path, required=True)
        q.add_argument("--encoder", required=True)
        q.add_argument("--out", type=Path, default=None, help="default: <names dir>/<encoder>")
        q.add_argument("--shard-size", type=int, default=SHARD_SIZE)
        q.add_argument("--batch-size", type=int, default=BATCH_SIZE)
        if name == "bench":
            q.add_argument("--n", type=int, default=500)
        q.set_defaults(func=cmd_shards)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
