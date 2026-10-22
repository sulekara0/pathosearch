"""Symmetric InfoNCE (CLIP loss) with a learnable temperature and a same-pair mask.

For a batch of N (image, text) pairs with L2-normalised embeddings, ``logits = s * I @ T^T``
where ``s = exp(log_scale)`` and ``log_scale`` is capped at ln(``MAX_LOGIT_SCALE``) (CLIP's 100). The loss is the
mean of the image->text and text->image cross-entropies with the diagonal as target.

``same[i, j]`` marks off-diagonal pairs that must not act as negatives (the same image, or
an identical caption): those logits are set to -inf in both directions. The diagonal of
``same`` is ignored. All arithmetic is float32 so fp16/bf16 inputs from autocast are safe.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from pathosearch.train.manifest import normalize_caption

MAX_LOGIT_SCALE = 100.0
CLIP_INIT_LOG_SCALE = math.log(100.0)  # pretrained CLIP logit_scale value


class LearnableTemperature(nn.Module):
    """``exp(log_scale)``, with ``log_scale`` capped at ``ln(MAX_LOGIT_SCALE)``.

    The cap is applied to the parameter (at construction and by :meth:`clamp_` after each
    optimizer step, as in OpenCLIP), not inside ``forward``: a clamp in the forward pass
    has zero gradient at the cap, and CLIP's pretrained value sits exactly there, which
    would freeze the temperature.
    """

    def __init__(self, init: float = CLIP_INIT_LOG_SCALE) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(float(init)))
        self.clamp_()

    @torch.no_grad()
    def clamp_(self) -> None:
        self.log_scale.clamp_(max=math.log(MAX_LOGIT_SCALE))

    def forward(self) -> torch.Tensor:
        return self.log_scale.float().exp()


def same_pair_mask(image_ids: Sequence[str], captions: Sequence[str]) -> torch.Tensor:
    """(N, N) bool: True where two batch items share an image id or a normalised caption."""
    if len(image_ids) != len(captions):
        raise ValueError("image_ids and captions differ in length")
    ids = list(image_ids)
    caps = [normalize_caption(c) for c in captions]
    n = len(ids)
    return torch.tensor(
        [[ids[i] == ids[j] or caps[i] == caps[j] for j in range(n)] for i in range(n)],
        dtype=torch.bool,
    )


def clip_infonce_terms(
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    scale: torch.Tensor,
    same: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Per-anchor cross-entropies ``{"i2t": (N,), "t2i": (N,)}``.

    ``scale`` is the (already clamped) logit scale, not its log.
    """
    if image_emb.shape != text_emb.shape or image_emb.ndim != 2:
        raise ValueError(
            f"embeddings must have the same shape (N, D): {image_emb.shape} vs {text_emb.shape}"
        )
    n = image_emb.shape[0]
    img = F.normalize(image_emb.float(), dim=-1)
    txt = F.normalize(text_emb.float(), dim=-1)
    logits = scale.float() * img @ txt.T
    if same is not None:
        if same.shape != (n, n):
            raise ValueError(f"mask must be ({n}, {n}), got {tuple(same.shape)}")
        off_diag = same.to(logits.device) & ~torch.eye(n, dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(off_diag, float("-inf"))
    target = torch.arange(n, device=logits.device)
    return {
        "i2t": F.cross_entropy(logits, target, reduction="none"),
        "t2i": F.cross_entropy(logits.T, target, reduction="none"),
    }


def clip_infonce(
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    scale: torch.Tensor,
    same: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scalar symmetric InfoNCE: mean of the image->text and text->image terms."""
    terms = clip_infonce_terms(image_emb, text_emb, scale, same)
    return (terms["i2t"].mean() + terms["t2i"].mean()) / 2
