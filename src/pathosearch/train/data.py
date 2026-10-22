"""Image-caption pairs for LoRA training.

One item per image per epoch; the caption comes from :func:`manifest.caption_for` (hash
rotation, no random state). Epoch order is a permutation seeded by (seed, epoch), so a
resumed run sees exactly the batches an uninterrupted run would. Missing images or
captions raise; nothing is substituted.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from pathosearch.errors import DataNotFoundError
from pathosearch.train.manifest import caption_for

MAX_TEXT_TOKENS = 77  # CLIP text context


def select_scale(rows: Sequence[Mapping[str, Any]], size: int) -> list[str]:
    """Images with ``scale_rank < size`` in rank order (scales are nested prefixes)."""
    ranks = sorted(int(r["scale_rank"]) for r in rows)
    if ranks != list(range(len(rows))):
        raise ValueError("scale_rank must be a permutation of 0..n-1")
    if size > len(rows):
        raise ValueError(f"scale of {size} requested, manifest has only {len(rows)} images")
    by_rank = sorted(rows, key=lambda r: int(r["scale_rank"]))
    return [str(r["image"]) for r in by_rank[:size]]


def write_captions_jsonl(path: Path, captions: Mapping[str, Sequence[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for name in sorted(captions):
            fh.write(json.dumps({"image": name, "captions": list(captions[name])}) + "\n")


def load_captions_jsonl(path: Path) -> dict[str, list[str]]:
    if not path.is_file():
        raise DataNotFoundError(f"Captions file not found: {path}")
    out: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                out[rec["image"]] = list(rec["captions"])
    return out


def epoch_order(n: int, seed: int, epoch: int) -> list[int]:
    g = torch.Generator().manual_seed(seed * 1_000_003 + epoch)
    return torch.randperm(n, generator=g).tolist()


@dataclass
class Item:
    image: Image.Image
    caption: str
    name: str


class PairDataset(Dataset[Item]):
    def __init__(
        self,
        names: Sequence[str],
        captions: Mapping[str, Sequence[str]],
        paths: Mapping[str, Path],
        caption_seed: int = 42,
    ) -> None:
        missing = [n for n in names if not captions.get(n)]
        if missing:
            raise ValueError(f"{len(missing)} images have no captions, e.g. {missing[:3]}")
        self.names = list(names)
        self.captions = {n: list(captions[n]) for n in self.names}
        self.paths = paths
        self.caption_seed = caption_seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, i: int) -> Item:
        name = self.names[i]
        path = self.paths.get(name)
        if path is None or not Path(path).is_file():
            raise DataNotFoundError(f"Image not found: {name}")
        with Image.open(path) as im:
            image = im.convert("RGB")
        caption = caption_for(name, self.captions[name], self.epoch, self.caption_seed)
        return Item(image, caption, name)


@dataclass
class Batch:
    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    names: list[str]
    captions: list[str]
    truncated: int  # captions longer than the text context (cut, not dropped)


class ClipCollator:
    """Turn items into model inputs with a ``CLIPProcessor`` (fixed-length padding)."""

    def __init__(self, processor: Any, max_length: int = MAX_TEXT_TOKENS) -> None:
        self.processor = processor
        self.max_length = max_length

    def __call__(self, items: Sequence[Item]) -> Batch:
        texts = [it.caption for it in items]
        # untruncated lengths, only to count cut captions (the count goes to run.json)
        lengths = [len(ids) for ids in self.processor.tokenizer(texts, verbose=False)["input_ids"]]
        enc = self.processor(
            images=[it.image for it in items],
            text=texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        return Batch(
            pixel_values=enc["pixel_values"],
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            names=[it.name for it in items],
            captions=texts,
            truncated=sum(1 for n in lengths if n > self.max_length),
        )
