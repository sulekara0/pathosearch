"""Training configuration (R4) and the small pure helpers the loop needs.

Defaults are the pre-registered R4 settings (ADR-0004); they are not tuned on the golden
set. ``config_hash`` identifies a run's settings in ``run.json`` and in eval reports.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from typing import Any

from pathosearch.train.model import BASE_MODEL, LoraSettings

AMP_MODES = ("auto", "off", "fp16", "bf16")


@dataclass(frozen=True)
class TrainConfig:
    name: str
    scale: str
    seed: int = 0
    base_model: str = BASE_MODEL
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_targets: tuple[str, ...] = ("q_proj", "v_proj")
    lr: float = 1e-4
    weight_decay: float = 0.1  # LoRA weights only; the temperature gets 0
    warmup_frac: float = 0.05
    epochs: int = 10
    batch_size: int = 256
    grad_clip: float = 1.0
    amp: str = "auto"
    num_workers: int = 2
    caption_seed: int = 42  # fixed: the caption schedule is the same for every training seed
    max_steps: int | None = None  # smoke runs only
    max_train_images: int | None = None  # smoke runs only
    max_val_images: int | None = None  # smoke runs only

    def __post_init__(self) -> None:
        if self.amp not in AMP_MODES:
            raise ValueError(f"amp must be one of {AMP_MODES}")
        if self.batch_size < 2:
            raise ValueError("contrastive training needs batch_size >= 2")

    def lora_settings(self) -> LoraSettings:
        return LoraSettings(self.lora_r, self.lora_alpha, self.lora_dropout, self.lora_targets)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["lora_targets"] = list(self.lora_targets)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config keys {sorted(unknown)}")
        args = dict(d)
        if "lora_targets" in args:
            args["lora_targets"] = tuple(args["lora_targets"])
        return cls(**args)


def config_hash(cfg: TrainConfig) -> str:
    blob = json.dumps(cfg.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def resolve_amp(amp: str, device: str, bf16_ok: bool) -> str:
    """``auto``: bf16 if supported, else fp16 on CUDA; off on CPU. Explicit modes are checked."""
    cuda = device.startswith("cuda")
    if amp == "auto":
        return ("bf16" if bf16_ok else "fp16") if cuda else "off"
    if amp in ("fp16", "bf16") and not cuda:
        raise ValueError(f"amp={amp} needs a CUDA device")
    if amp == "bf16" and not bf16_ok:
        raise ValueError("amp=bf16 is not supported on this GPU")
    return amp


def lr_factor(step: int, total: int, warmup: int) -> float:
    """Linear warmup over ``warmup`` steps (step 0 -> 1/warmup), then cosine to 0 at ``total``."""
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    span = max(1, total - warmup)
    progress = min(1.0, (step - warmup) / span)
    return 0.5 * (1.0 + math.cos(math.pi * progress))
