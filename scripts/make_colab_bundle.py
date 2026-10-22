"""Thin CLI wrapper: see pathosearch.train.bundle.

Usage (from the repo root; paths come from .env / PATHOSEARCH_* variables):
    uv run python scripts/make_colab_bundle.py --manifest eval_data/r4_train_v0

Writes ``artifacts/r4_bundle/<manifest name>.zip`` (all train + val images, short side 224,
plus captions.jsonl) and ``<same>.json`` with counts, sha256 and the command. Upload the zip
to Google Drive; it is never committed (Quilt-1M data, private use only).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from pathosearch.config import load_settings, require_file
from pathosearch.eval.labeling import build_image_index
from pathosearch.eval.run_eval import git_sha
from pathosearch.train.bundle import BUNDLE_SIZE, JPEG_QUALITY, build_bundle
from pathosearch.train.manifest import load_image_meta
from pathosearch.train.runner import file_sha256, load_manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--size", type=int, default=BUNDLE_SIZE)
    p.add_argument("--quality", type=int, default=JPEG_QUALITY)
    p.add_argument("--limit", type=int, default=None, help="first N names only (trial bundle)")
    args = p.parse_args()

    s = load_settings()
    m = load_manifest(args.manifest)
    names = sorted({r["image"] for r in m.train_rows} | set(m.val_names))
    if args.limit is not None:
        names = names[: args.limit]
    suffix = f"_first{args.limit}" if args.limit is not None else ""
    out = args.out or s.artifacts_dir / "r4_bundle" / f"{args.manifest.name}{suffix}.zip"

    csv = require_file(s.csv_path, "Set PATHOSEARCH_CSV_PATH in .env")
    print(f"{len(names)} images; reading captions from the CSV ...")
    meta = load_image_meta(csv, names)
    result = build_bundle(
        names,
        build_image_index(s.images_dir),
        {n: meta[n].captions for n in names},
        out,
        args.size,
        args.quality,
    )
    info = {
        **result,
        "manifest": args.manifest.as_posix(),
        "train_manifest_sha256": file_sha256(args.manifest / "train_manifest.jsonl"),
        "val_manifest_sha256": file_sha256(args.manifest / "val_manifest.jsonl"),
        "limit": args.limit,
        "git_sha": git_sha(),
        "command": " ".join(["uv run python scripts/make_colab_bundle.py", *sys.argv[1:]]),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    mb = result["zip_bytes"] / 1e6
    print(f"wrote {out} ({mb:.0f} MB, sha256 {result['zip_sha256'][:12]}...)")


if __name__ == "__main__":
    main()
