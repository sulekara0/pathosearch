"""Synthetic golden directory + images + caption CSV for trying the labeling UI offline.

Everything is generated from a seed; nothing here touches real data. The fake images are
colored noise with their file name drawn on them, so a wrong image/label pairing is visible.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from pathosearch.eval.golden import (
    CorpusEntry,
    GoldenPaths,
    GoldenQuery,
    save_corpus,
    save_queries,
)
from pathosearch.eval.pooling import build_pool, overlap_subset, save_pool

CATEGORIES = ("dermatopathology", "gastrointestinal", "pulmonary", "soft tissue", "unknown")
QUERY_TEXTS = (
    "fake query: squamous pattern",
    "fake query: glandular structures",
    "fake query: alveolar spaces",
    "fake query: spindle cell bundles",
    "fake query: keratin pearls",
    "fake query: mucin pools",
)
SYSTEMS = ("sys1", "sys2")


@dataclass(frozen=True)
class FakeGolden:
    golden_dir: Path
    images_dir: Path
    csv_path: Path

    def run_command(self, annotator: str = "A") -> str:
        return (
            f"uv run streamlit run label_app.py -- --golden {self.golden_dir.as_posix()} "
            f"--images-dir {self.images_dir.as_posix()} --csv {self.csv_path.as_posix()} "
            f"--annotator {annotator}"
        )


def _fake_image(
    name: str, rng: np.random.Generator, size: tuple[int, int] = (640, 480)
) -> Image.Image:
    w, h = size
    base = rng.integers(0, 256, size=3)
    noise = rng.integers(-40, 40, size=(h, w, 3))
    arr = np.clip(base + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    ImageDraw.Draw(img).text((12, 12), name, fill=(255, 255, 255))
    return img


def make_fake_golden(
    out_dir: Path,
    n_images: int = 60,
    n_queries: int = 6,
    depth: int = 5,
    n_random: int = 2,
    seed: int = 42,
) -> FakeGolden:
    """Write ``out_dir/{golden,images,captions.csv}`` and return their locations."""
    if not 1 <= n_queries <= len(QUERY_TEXTS):
        raise ValueError(f"n_queries must be in 1..{len(QUERY_TEXTS)}")
    if n_images < depth * len(SYSTEMS) + n_random:
        raise ValueError("n_images too small for the requested pool depth")
    rng = np.random.default_rng(seed)
    golden = GoldenPaths(out_dir / "golden")
    images_dir = out_dir / "images"
    csv_path = out_dir / "captions.csv"

    names = [f"fake_{i:04d}.jpg" for i in range(n_images)]
    entries = [
        CorpusEntry(n, CATEGORIES[i % len(CATEGORIES)], "val_only") for i, n in enumerate(names)
    ]
    for i, name in enumerate(names):
        sub = images_dir / f"part_{i % 2 + 1}"  # nested dirs: image lookup must be recursive
        sub.mkdir(parents=True, exist_ok=True)
        _fake_image(name, rng).save(sub / name, quality=90)

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_path", "caption", "pathology", "split"])
        for e in entries:
            writer.writerow([e.image, f"fake first caption of {e.image}", e.pathology, "val"])
            if int(e.image[5:9]) % 3 == 0:  # some images carry a second caption
                writer.writerow([e.image, f"fake second caption of {e.image}", e.pathology, "val"])

    queries = [
        GoldenQuery(f"q{i + 1:03d}", QUERY_TEXTS[i], CATEGORIES[i % len(CATEGORIES)], "fake")
        for i in range(n_queries)
    ]
    runs = {
        system: {q.qid: [names[j] for j in rng.permutation(n_images)[:depth]] for q in queries}
        for system in SYSTEMS
    }
    pool = build_pool(runs, names, depth=depth, n_random=n_random, seed=seed)

    save_corpus(golden.corpus, entries)
    save_queries(golden.queries, queries)
    save_pool(golden.pool, pool)
    save_pool(golden.overlap, overlap_subset(pool, seed=seed))
    return FakeGolden(golden.root, images_dir, csv_path)
