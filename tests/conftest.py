"""Shared fixtures: a tiny random CLIP saved to disk (no downloads) and fake images."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

TINY_IMAGE = 32
TINY_VOCAB = 100


def tiny_clip_config():
    from transformers import CLIPConfig

    return CLIPConfig(
        text_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "vocab_size": TINY_VOCAB,
            "max_position_embeddings": 16,
            "bos_token_id": 0,
            "eos_token_id": 1,
            "pad_token_id": 1,
        },
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "image_size": TINY_IMAGE,
            "patch_size": 8,
        },
        projection_dim=16,
    )


@pytest.fixture
def tiny_base_dir(tmp_path: Path) -> Path:
    """A random tiny CLIPModel saved with ``save_pretrained``; logit_scale set to 2.5."""
    from transformers import CLIPModel

    torch.manual_seed(0)
    model = CLIPModel(tiny_clip_config())
    with torch.no_grad():
        model.logit_scale.fill_(2.5)
    d = tmp_path / "base"
    model.save_pretrained(d)
    return d


def make_image_files(root: Path, names: list[str]) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    out = {}
    for i, n in enumerate(names):
        p = root / n
        Image.new("RGB", (40, 36), ((37 * i) % 256, (91 * i) % 256, (53 * i) % 256)).save(p)
        out[n] = p
    return out
