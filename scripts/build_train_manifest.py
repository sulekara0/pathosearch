"""Thin CLI wrapper: see pathosearch.train.manifest.

Usage (from the repo root; paths come from .env / PATHOSEARCH_* variables):
    uv run python scripts/build_train_manifest.py --golden eval_data/golden_v0 \
        --out eval_data/r4_train_v0

Writes ``train_manifest.jsonl``, ``val_manifest.jsonl`` and ``train_manifest_meta.json``
(image names and counts only, no captions) and ``reports/train_manifest_<name>.md``.
Refuses to overwrite an existing manifest directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pathosearch.config import load_settings, require_dir, require_file
from pathosearch.eval.corpus_analysis import list_image_names, scan_csv, split_class
from pathosearch.eval.golden import GoldenPaths, read_jsonl, write_jsonl
from pathosearch.eval.run_eval import git_sha
from pathosearch.train.manifest import (
    NEAR_DUP_TAU,
    SEED,
    VAL_SIZE,
    exclude,
    load_image_meta,
    make_split,
    render_markdown,
    summary,
    train_rows,
    val_rows,
)

ENCODERS = ("clip", "quilt")


def file_info(path: Path, digest: bool = False) -> dict[str, Any]:
    st = path.stat()
    info: dict[str, Any] = {
        "path": path.as_posix(),
        "size_bytes": st.st_size,
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(
            timespec="seconds"
        ),
    }
    if digest:
        info["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return info


def load_ids(path: Path) -> list[str]:
    return list(json.loads(require_file(path).read_text(encoding="utf-8")))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--golden", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--val-size", type=int, default=VAL_SIZE)
    p.add_argument("--tau", type=float, default=NEAR_DUP_TAU)
    args = p.parse_args()
    if args.out.exists():
        sys.exit(f"{args.out} exists; a manifest is never overwritten (use a new name)")

    s = load_settings()
    golden = GoldenPaths(args.golden)
    cand_root = s.artifacts_dir / f"{golden.root.name}_candidates"

    main_ids = load_ids(s.encoder_dir("clip") / "ids.json")
    if load_ids(s.encoder_dir("quilt") / "ids.json") != main_ids:
        sys.exit("clip and quilt main-corpus ids differ; embeddings cannot be paired")
    main_emb = {e: np.load(s.encoder_dir(e) / "embeddings.npy", mmap_mode="r") for e in ENCODERS}

    golden_ids = load_ids(cand_root / "clip" / "ids.json")
    golden_emb: dict[str, np.ndarray] = {}
    for e in ENCODERS:
        ids = load_ids(cand_root / e / "ids.json")
        emb = np.load(cand_root / e / "embeddings.npy")
        pos = {n: i for i, n in enumerate(ids)}
        golden_emb[e] = emb[[pos[n] for n in golden_ids]]
    nearest_path = golden.root / "excluded_near_dup.jsonl"
    nearest = {
        enc["nearest"]
        for rec in read_jsonl(require_file(nearest_path))
        for enc in rec["encoders"].values()
    }

    disk, _ = list_image_names(require_dir(s.images_dir, "Set PATHOSEARCH_IMAGES_DIR in .env"))
    scan = scan_csv(require_file(s.csv_path, "Set PATHOSEARCH_CSV_PATH in .env"), set())
    candidates = sorted(
        n for n in disk if n in scan.image_mask and split_class(scan.image_mask[n]) == "train_only"
    )
    main_set = set(main_ids)
    print(f"candidates {len(candidates)}; loading CSV rows for those with embeddings ...")
    meta = load_image_meta(s.csv_path, [n for n in candidates if n in main_set] + golden_ids)

    excl = exclude(
        candidates, main_ids, golden_ids, nearest, main_emb, main_ids, golden_emb, meta, args.tau
    )
    split = make_split(excl.kept, main_emb, main_ids, args.val_size, args.seed, args.tau)

    out_meta = {
        "name": args.out.name,
        "command": " ".join(["uv run python scripts/build_train_manifest.py", *sys.argv[1:]]),
        "git_sha": git_sha(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "near_dup_tau": args.tau,
        "near_dup_encoders": list(ENCODERS),
        "val_size": args.val_size,
        "inputs": {
            "csv": file_info(s.csv_path),
            "images_dir": s.images_dir.as_posix(),
            "main_ids": file_info(s.encoder_dir("clip") / "ids.json"),
            "golden_corpus": file_info(golden.corpus, digest=True),
            "golden_candidates_ids": file_info(cand_root / "clip" / "ids.json", digest=True),
            "golden_excluded_near_dup": file_info(nearest_path, digest=True),
        },
        **summary(excl, split, meta, len(candidates)),
    }
    args.out.mkdir(parents=True)
    write_jsonl(args.out / "train_manifest.jsonl", train_rows(split, meta))
    write_jsonl(args.out / "val_manifest.jsonl", val_rows(split, meta))
    (args.out / "train_manifest_meta.json").write_text(
        json.dumps(out_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = Path("reports") / f"train_manifest_{args.out.name}.md"
    report.write_text(render_markdown(out_meta), encoding="utf-8")
    print(f"wrote {args.out} and {report}")


if __name__ == "__main__":
    main()
