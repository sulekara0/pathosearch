"""LoRA on a Hugging Face CLIP: adapters on attention q/v in both towers, plus a temperature.

Everything of the base model is frozen (projections, LayerNorms, embeddings and the base
``logit_scale``). Trainable: the LoRA A/B matrices and :class:`LearnableTemperature`,
initialised from the base ``logit_scale``. LoRA B starts at zero, so the model equals the
base model before training.

An adapter directory holds the peft adapter files and ``lora_meta.json`` (base model id,
LoRA settings, learned log-scale, package versions); the base weights are not copied.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from transformers import CLIPModel

from pathosearch.config import require_file
from pathosearch.train.loss import LearnableTemperature

BASE_MODEL = "openai/clip-vit-base-patch32"
META_FILE = "lora_meta.json"


@dataclass(frozen=True)
class LoraSettings:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.1
    targets: tuple[str, ...] = ("q_proj", "v_proj")


def _pooled(out: Any) -> torch.Tensor:
    # transformers >= 5 returns BaseModelOutputWithPooling instead of a tensor
    return out if isinstance(out, torch.Tensor) else out.pooler_output


class LoraCLIP(nn.Module):
    """CLIP with LoRA adapters (or merged weights) and a learnable temperature.

    ``encode_*`` return projected, **unnormalised** features; the loss normalises.
    """

    def __init__(self, clip: nn.Module, temperature: LearnableTemperature, meta: dict[str, Any]):
        super().__init__()
        # PeftModel (training) or merged CLIPModel (inference); both expose get_*_features
        self.clip: Any = clip
        self.temperature = temperature
        self.lora_meta = meta

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return _pooled(self.clip.get_image_features(pixel_values=pixel_values))

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return _pooled(
            self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        )


def _lora_config(s: LoraSettings) -> LoraConfig:
    return LoraConfig(
        r=s.r,
        lora_alpha=s.alpha,
        lora_dropout=s.dropout,
        target_modules=list(s.targets),
        bias="none",
    )


def build_lora_clip(base_model: str = BASE_MODEL, settings: LoraSettings | None = None) -> LoraCLIP:
    """Load ``base_model``, freeze it and attach fresh LoRA adapters."""
    s = settings or LoraSettings()
    base = CLIPModel.from_pretrained(base_model)
    init = float(base.logit_scale.detach().item())
    for p in base.parameters():
        p.requires_grad_(False)
    peft_model = get_peft_model(base, _lora_config(s))
    meta = {"base_model": base_model, "lora": asdict(s)}
    return LoraCLIP(peft_model, LearnableTemperature(init), meta)


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [n for n, p in model.named_parameters() if p.requires_grad]


def count_parameters(model: nn.Module) -> dict[str, int]:
    params = list(model.parameters())
    return {
        "trainable": sum(p.numel() for p in params if p.requires_grad),
        "total": sum(p.numel() for p in params),
    }


def _versions() -> dict[str, str]:
    return {n: metadata.version(n) for n in ("torch", "transformers", "peft")}


def save_lora_clip(model: LoraCLIP, out_dir: Path) -> None:
    """Write the adapter and ``lora_meta.json`` (base weights are not saved)."""
    if not isinstance(model.clip, PeftModel):
        raise TypeError("only an unmerged LoRA model can be saved as an adapter")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.clip.save_pretrained(str(out_dir))
    meta = {
        **model.lora_meta,
        "log_scale": float(model.temperature.log_scale.detach().item()),
        "versions": _versions(),
    }
    (out_dir / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_lora_clip(adapter_dir: Path, merge: bool = False) -> LoraCLIP:
    """Rebuild base + adapter. ``merge=True`` folds LoRA into the weights (inference only)."""
    meta_path = require_file(Path(adapter_dir) / META_FILE, "Not a LoRA adapter directory")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    base = CLIPModel.from_pretrained(meta["base_model"])
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
    clip: nn.Module = peft_model.merge_and_unload() if merge else peft_model
    for p in clip.parameters():
        p.requires_grad_(False)
    temperature = LearnableTemperature(meta["log_scale"]).requires_grad_(False)
    model = LoraCLIP(clip, temperature, meta)
    return model.eval()
