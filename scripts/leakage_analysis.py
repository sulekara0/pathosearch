"""Thin CLI (B3 follow-up): descriptive analysis of the golden-vs-main near-duplicate cosines.

Measures only. It does not change the leakage threshold or the leakage code.
Writes reports/leakage_analysis_<golden>.{json,md} and a side-by-side HTML for the
cosine >= --pair_threshold pairs.

Usage:
    uv run python scripts/leakage_analysis.py --golden eval_data/golden_v0
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from pathosearch.config import load_settings, require_dir, require_file
from pathosearch.eval.golden import GoldenPaths, load_corpus
from pathosearch.eval.leakage_analysis import clusters, distribution, nearest_cosine, within_pairs
from pathosearch.eval.run_eval import command_line
from pathosearch.faiss_search import load_embeddings

TAU = 0.95  # the pre-registered leakage threshold, only used to describe within-golden pairs


def csv_captions(csv_path: Path, names: set[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for chunk in pd.read_csv(csv_path, usecols=["image_path", "caption"], chunksize=200_000):
        hit = chunk[chunk["image_path"].isin(names)][["image_path", "caption"]]
        for n, c in hit.itertuples(index=False):
            out.setdefault(n, []).append(str(c))
    return out


def find_images(root: Path, names: set[str]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for d, _, files in os.walk(root):
        for f in files:
            if f in names:
                found[f] = Path(d) / f
    return found


def thumb(path: Path | None) -> str:
    if path is None:
        return '<div class="missing">image not on disk</div>'
    im = Image.open(path).convert("RGB")
    im.thumbnail((280, 280))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    return f'<img src="data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--golden", type=Path, required=True)
    p.add_argument("--encoders", default="clip,quilt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pair_threshold", type=float, default=0.9999)
    p.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = p.parse_args()

    s = load_settings()
    golden = GoldenPaths(args.golden)
    gname = golden.root.name
    golden_ids = [e.image for e in load_corpus(golden.corpus)]
    rng = np.random.default_rng(args.seed)

    res: dict[str, Any] = {}
    pair_rows: list[dict[str, Any]] = []
    flagged: dict[str, set[str]] = {}
    for enc in args.encoders.split(","):
        g_emb, g_ids = load_embeddings(str(require_dir(s.artifacts_dir / gname / enc)))
        m_emb, m_ids = load_embeddings(str(require_dir(s.encoder_dir(enc))))
        if g_ids != golden_ids and set(g_ids) != set(golden_ids):
            raise SystemExit(f"{enc}: golden embeddings do not match the manifest")
        best, arg = nearest_cosine(g_emb, m_emb)
        sample = np.sort(rng.choice(len(m_ids), size=len(g_ids), replace=False))
        ref_best, _ = nearest_cosine(m_emb[sample], m_emb, exclude_self_rows=sample.tolist())
        flagged[enc] = {g_ids[i] for i in np.nonzero(best > TAU)[0]}
        g_pairs = within_pairs(g_emb, g_ids, TAU)
        g_pairs99 = within_pairs(g_emb, g_ids, 0.99)
        res[enc] = {
            "golden_vs_main": distribution(best),
            "random_main_vs_rest_of_main": distribution(ref_best),
            "within_golden_tau_0.95": clusters(g_pairs),
            "within_golden_tau_0.99": clusters(g_pairs99),
        }
        for i in np.nonzero(best >= args.pair_threshold)[0]:
            pair_rows.append(
                {
                    "encoder": enc,
                    "golden": g_ids[i],
                    "nearest": m_ids[int(arg[i])],
                    "cosine": round(float(best[i]), 6),
                }
            )

    res["flagged_distinct_images_tau_0.95"] = {
        "per_encoder": {e: len(v) for e, v in flagged.items()},
        "union": len(set().union(*flagged.values())),
        "intersection": len(set.intersection(*map(set, flagged.values()))),
    }

    names = {r["golden"] for r in pair_rows} | {r["nearest"] for r in pair_rows}
    caps = csv_captions(require_file(s.csv_path, "Set PATHOSEARCH_CSV_PATH in .env"), names)
    imgs = find_images(s.images_dir, names)
    for r in pair_rows:
        r["golden_captions"] = caps.get(r["golden"], [])
        r["nearest_captions"] = caps.get(r["nearest"], [])
        r["golden_on_disk"] = r["golden"] in imgs
        r["nearest_on_disk"] = r["nearest"] in imgs
        r["captions_identical"] = bool(r["golden_captions"]) and set(r["golden_captions"]) == set(
            r["nearest_captions"]
        )
    res["pairs_ge_threshold"] = {"threshold": args.pair_threshold, "pairs": pair_rows}
    res["meta"] = {"command": command_line(), "seed": args.seed, "tau_for_description": TAU}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jpath = args.out_dir / f"leakage_analysis_{gname}.json"
    jpath.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    cards = []
    for r in pair_rows:
        gc = " | ".join(html.escape(c[:300]) for c in r["golden_captions"]) or "(none)"
        nc = " | ".join(html.escape(c[:300]) for c in r["nearest_captions"]) or "(none)"
        cards.append(
            f"<div class=row><h3>{r['encoder']} cos={r['cosine']}</h3><div class=pair>"
            f"<figure>{thumb(imgs.get(r['golden']))}<figcaption><b>golden</b> "
            f"{html.escape(r['golden'])}<br>{gc}</figcaption></figure>"
            f"<figure>{thumb(imgs.get(r['nearest']))}<figcaption><b>main</b> "
            f"{html.escape(r['nearest'])}<br>{nc}</figcaption></figure></div></div>"
        )
    hpath = args.out_dir / f"leakage_pairs_{gname}.html"
    hpath.write_text(
        "<!doctype html><meta charset=utf-8><title>near-duplicate pairs</title>"
        "<style>body{font:14px sans-serif;margin:16px}.pair{display:flex;gap:16px}"
        "figure{margin:0;max-width:320px}.missing{width:280px;height:120px;background:#eee;"
        "display:flex;align-items:center;justify-content:center}</style>"
        f"<h1>Pairs with cosine &ge; {args.pair_threshold}</h1>{''.join(cards)}",
        encoding="utf-8",
    )
    print(f"wrote {jpath} and {hpath}")


if __name__ == "__main__":
    main()
