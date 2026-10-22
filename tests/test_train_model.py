"""LoRA CLIP: what is trainable, identity at init, frozen base, save/load round trip.

A tiny random CLIP is saved to a temp dir and used as the "pretrained" base, so no
weights are downloaded.
"""

from pathlib import Path

import pytest
import torch
from transformers import CLIPConfig, CLIPModel

from pathosearch.train.model import (
    LoraSettings,
    build_lora_clip,
    load_lora_clip,
    save_lora_clip,
    trainable_parameter_names,
)

TARGETS = ("q_proj", "v_proj")


def tiny_config() -> CLIPConfig:
    return CLIPConfig(
        text_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "vocab_size": 100,
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
            "image_size": 32,
            "patch_size": 8,
        },
        projection_dim=16,
    )


@pytest.fixture
def base_dir(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    model = CLIPModel(tiny_config())
    with torch.no_grad():
        model.logit_scale.fill_(2.5)  # distinct from the default so the init is checked
    d = tmp_path / "base"
    model.save_pretrained(d)
    return d


def batch(n: int = 3) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(1)
    return {
        "pixel_values": torch.randn(n, 3, 32, 32, generator=g),
        "input_ids": torch.randint(2, 100, (n, 8), generator=g),
        "attention_mask": torch.ones(n, 8, dtype=torch.long),
    }


def embed(model, b: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        return model.encode_image(b["pixel_values"]), model.encode_text(
            b["input_ids"], b["attention_mask"]
        )


def base_embed(base_dir: Path, b: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    base = CLIPModel.from_pretrained(base_dir).eval()
    with torch.no_grad():
        img = base.get_image_features(pixel_values=b["pixel_values"])
        txt = base.get_text_features(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
    if not isinstance(img, torch.Tensor):
        img, txt = img.pooler_output, txt.pooler_output
    return img, txt


def perturb_lora(model) -> None:
    """Stand-in for training: make every LoRA B matrix non-zero."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(std=0.5)


def test_only_lora_and_temperature_are_trainable(base_dir: Path) -> None:
    model = build_lora_clip(str(base_dir), LoraSettings(r=4, alpha=8))
    names = trainable_parameter_names(model)
    assert "temperature.log_scale" in names
    lora = [n for n in names if n != "temperature.log_scale"]
    assert lora and all("lora_A" in n or "lora_B" in n for n in lora)
    assert all(any(t in n for t in TARGETS) for n in lora)
    assert not any(k in n for n in lora for k in ("k_proj", "out_proj", "mlp", "projection"))
    for tower in ("vision_model", "text_model"):
        for layer in range(2):
            for t in TARGETS:
                assert any(f"{tower}.encoder.layers.{layer}.self_attn.{t}" in n for n in lora)
    # base logit_scale is replaced by the temperature module, not trained twice
    assert not any(n.endswith("logit_scale") for n in names)


def test_identity_at_init(base_dir: Path) -> None:
    b = batch()
    img, txt = embed(build_lora_clip(str(base_dir), LoraSettings(r=4, alpha=8)), b)
    ref_img, ref_txt = base_embed(base_dir, b)
    assert torch.allclose(img, ref_img, atol=1e-6) and torch.allclose(txt, ref_txt, atol=1e-6)


def test_temperature_starts_from_base_logit_scale(base_dir: Path) -> None:
    model = build_lora_clip(str(base_dir), LoraSettings(r=4, alpha=8))
    assert model.temperature.log_scale.item() == pytest.approx(2.5)


def test_optimizer_step_leaves_base_weights_unchanged(base_dir: Path) -> None:
    from pathosearch.train.loss import clip_infonce

    model = build_lora_clip(str(base_dir), LoraSettings(r=4, alpha=8))
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    b = batch(4)
    model.train()
    for _ in range(2):
        loss = clip_infonce(
            model.encode_image(b["pixel_values"]),
            model.encode_text(b["input_ids"], b["attention_mask"]),
            model.temperature(),
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    for n, p in model.named_parameters():
        if n in frozen:
            assert torch.equal(p, frozen[n]), n
    # LoRA B starts at zero; after steps it must have moved
    assert any(p.abs().sum() > 0 for n, p in model.named_parameters() if "lora_B" in n)


def test_save_load_round_trip(base_dir: Path, tmp_path: Path) -> None:
    settings = LoraSettings(r=4, alpha=8)
    model = build_lora_clip(str(base_dir), settings)
    perturb_lora(model)
    with torch.no_grad():
        model.temperature.log_scale.fill_(3.0)
    b = batch()
    before = embed(model, b)
    assert not torch.allclose(before[0], base_embed(base_dir, b)[0], atol=1e-4)

    out = tmp_path / "adapter"
    save_lora_clip(model, out)
    for merge in (False, True):
        loaded = load_lora_clip(out, merge=merge)
        after = embed(loaded, b)
        assert torch.allclose(after[0], before[0], atol=1e-5), merge
        assert torch.allclose(after[1], before[1], atol=1e-5), merge
        assert loaded.temperature.log_scale.item() == pytest.approx(3.0)
    meta = loaded.lora_meta
    assert meta["base_model"] == str(base_dir) and meta["lora"]["r"] == 4


def test_eval_mode_is_deterministic_despite_dropout(base_dir: Path) -> None:
    model = build_lora_clip(str(base_dir), LoraSettings(r=4, alpha=8, dropout=0.5))
    perturb_lora(model)
    b = batch()
    assert torch.equal(embed(model, b)[0], embed(model, b)[0])


def test_load_requires_meta(tmp_path: Path) -> None:
    (tmp_path / "x").mkdir()
    with pytest.raises(FileNotFoundError):
        load_lora_clip(tmp_path / "x")
