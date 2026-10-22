"""How much of the training text is low-information (ADR-0004 decision input).

Two indicators, both on normalised captions (:func:`manifest.normalize_caption`):
short captions (< ``short_chars`` characters) and the most frequent captions (by number of
images carrying them). ``epoch0_short`` counts images whose epoch-0 caption (the one the
rotation picks first) is short. Only counts are returned unless ``preview_chars > 0``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from pathosearch.train.manifest import SEED, caption_for, normalize_caption

SHORT_CHARS = 40
TOP_N = 20
PREVIEW_CHARS = 0  # caption text stays out of committed reports unless asked for


def caption_stats(
    captions: Mapping[str, Sequence[str]],
    short_chars: int = SHORT_CHARS,
    top_n: int = TOP_N,
    caption_seed: int = SEED,
    preview_chars: int = PREVIEW_CHARS,
) -> dict[str, Any]:
    norm = {n: [normalize_caption(c) for c in caps] for n, caps in captions.items()}

    def short(c: str) -> bool:
        return len(c) < short_chars

    per_image = Counter(c for caps in norm.values() for c in set(caps))
    top = per_image.most_common(top_n)
    top_set = {c for c, _ in top}
    return {
        "images": len(norm),
        "captions": sum(len(c) for c in norm.values()),
        "short_chars": short_chars,
        "short_captions": sum(short(c) for caps in norm.values() for c in caps),
        "images_any_short": sum(any(map(short, caps)) for caps in norm.values()),
        "images_all_short": sum(bool(caps) and all(map(short, caps)) for caps in norm.values()),
        "epoch0_short": sum(
            short(normalize_caption(caption_for(n, list(c), 0, caption_seed)))
            for n, c in captions.items()
            if c
        ),
        "top_n": top_n,
        "top_images": [k for _, k in top],
        **({"top_preview": [c[:preview_chars] for c, _ in top]} if preview_chars > 0 else {}),
        "images_any_top": sum(any(c in top_set for c in caps) for caps in norm.values()),
        "images_all_top": sum(
            bool(caps) and all(c in top_set for c in caps) for caps in norm.values()
        ),
    }
