"""LoRA encoder: same vectors as the trained model, L2-normalised, reachable via get_encoder."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from conftest import TINY_IMAGE, TINY_VOCAB
from pathosearch import encoders
from pathosearch.encoders import LoRACLIPEncoder, get_encoder
from pathosearch.train.model import LoraSettings, build_lora_clip, save_lora_clip


class FakeProcessor:
    """Minimal CLIPProcessor stand-in at the tiny model's sizes."""

    def __call__(self, images=None, text=None, **kw):
        out = {}
        if images is not None:
            out["pixel_values"] = torch.stack(
                [
                    torch.from_numpy(
                        np.asarray(im.resize((TINY_IMAGE, TINY_IMAGE)), np.float32) / 255.0
                    ).permute(2, 0, 1)
                    for im in images
                ]
            )
        if text is not None:
            ids = torch.tensor(
                [
                    [0] + [2 + (ord(c) % (TINY_VOCAB - 2)) for c in t[:6].ljust(6)] + [1]
                    for t in text
                ]
            )
            out["input_ids"], out["attention_mask"] = ids, torch.ones_like(ids)
        return out


@pytest.fixture
def adapter(tiny_base_dir: Path, tmp_path: Path):
    model = build_lora_clip(str(tiny_base_dir), LoraSettings(r=4, alpha=8))
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_(std=0.5)
    out = tmp_path / "adapter"
    save_lora_clip(model, out, {"epoch": 3})
    return model.eval(), out


def images() -> list[Image.Image]:
    return [Image.new("RGB", (40, 40), c) for c in ((200, 10, 10), (10, 200, 10))]


def test_lora_encoder_matches_trained_model(adapter) -> None:
    model, out = adapter
    enc = LoRACLIPEncoder(out, device="cpu", processor=FakeProcessor())
    proc = FakeProcessor()
    with torch.no_grad():
        ref_img = model.encode_image(proc(images=images())["pixel_values"])
        t = proc(text=["squamous", "glands"])
        ref_txt = model.encode_text(t["input_ids"], t["attention_mask"])
    got_img, got_txt = enc.encode_images(images()), enc.encode_texts(["squamous", "glands"])
    assert got_img.dtype == np.float32 and got_img.shape == (2, 16)
    assert np.allclose(np.linalg.norm(got_img, axis=1), 1.0, atol=1e-5)
    assert np.allclose(got_img, torch.nn.functional.normalize(ref_img, dim=-1).numpy(), atol=1e-5)
    assert np.allclose(got_txt, torch.nn.functional.normalize(ref_txt, dim=-1).numpy(), atol=1e-5)
    assert enc.embed_dim == 16 and enc.name == f"lora:{out.as_posix()}"
    assert enc.info()["epoch"] == 3


def test_get_encoder_routes_lora_prefix(adapter, monkeypatch: pytest.MonkeyPatch) -> None:
    _, out = adapter
    monkeypatch.setattr(encoders, "_clip_processor", lambda base: FakeProcessor())
    enc = get_encoder(f"lora:{out}", device="cpu")
    assert isinstance(enc, LoRACLIPEncoder)


def test_lora_encoder_missing_adapter_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        LoRACLIPEncoder(tmp_path / "nope", device="cpu", processor=FakeProcessor())
