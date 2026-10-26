"""R4 internal-validation baseline: the untrained base CLIP (no LoRA) scored exactly as the
training loop scores each epoch (:func:`pathosearch.train.loop.evaluate`).

Same val manifest, same caption schedule (``caption_seed``, epoch 0), same loss and recall
code; only the model differs. The temperature is the base model's ``logit_scale``, which is
also what the LoRA run starts from, so the numbers are the "epoch 0" point of a run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from pathosearch.train.config import TrainConfig
from pathosearch.train.data import PairDataset
from pathosearch.train.loop import Collate, evaluate
from pathosearch.train.loss import LearnableTemperature
from pathosearch.train.model import BASE_MODEL, LoraCLIP


def build_base_clip(base_model: str = BASE_MODEL) -> LoraCLIP:
    """Frozen base CLIP in the training wrapper, without adapters."""
    from transformers import CLIPModel

    base = CLIPModel.from_pretrained(base_model)
    for p in base.parameters():
        p.requires_grad_(False)
    temperature = LearnableTemperature(float(base.logit_scale.item())).requires_grad_(False)
    return LoraCLIP(base, temperature, {"base_model": base_model, "lora": None}).eval()


def evaluate_base(
    model: LoraCLIP,
    cfg: TrainConfig,
    val_names: Sequence[str],
    captions: Mapping[str, Sequence[str]],
    paths: Mapping[str, Path],
    collate: Collate,
    device: str = "cpu",
) -> dict[str, float]:
    """Validation metrics for ``model`` with the run's caption seed and batch size."""
    ds = PairDataset(val_names, captions, paths, cfg.caption_seed)
    return evaluate(model, ds, collate, device, "off", cfg.batch_size)
