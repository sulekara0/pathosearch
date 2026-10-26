"""Base-CLIP validation baseline on a tiny random CLIP (CPU, no downloads)."""

import math
from pathlib import Path

import pytest

from conftest import make_image_files
from pathosearch.eval.val_baseline import build_base_clip, evaluate_base
from pathosearch.train.config import TrainConfig
from pathosearch.train.data import PairDataset
from pathosearch.train.loop import evaluate
from pathosearch.train.model import build_lora_clip, count_parameters
from test_train_loop import fake_collate

VAL = [f"v{i}.jpg" for i in range(6)]


def test_base_clip_has_no_adapters_and_base_temperature(tiny_base_dir: Path) -> None:
    model = build_base_clip(str(tiny_base_dir))
    assert count_parameters(model)["trainable"] == 0
    assert not any("lora" in n for n, _ in model.named_parameters())
    assert model.lora_meta["lora"] is None and not model.training
    assert float(model.temperature()) == pytest.approx(math.exp(2.5))


def test_baseline_equals_untrained_lora_through_the_loop_evaluate(
    tiny_base_dir: Path, tmp_path: Path
) -> None:
    paths = make_image_files(tmp_path / "imgs", VAL)
    caps = {n: [f"caption {n}", f"other text {n}"] for n in VAL}
    cfg = TrainConfig(name="t", scale="full", base_model=str(tiny_base_dir), batch_size=4)

    base = evaluate_base(build_base_clip(str(tiny_base_dir)), cfg, VAL, caps, paths, fake_collate)
    assert set(base) == {"val_loss", "i2t_r1", "t2i_r1", "i2t_r5", "t2i_r5"}
    assert math.isfinite(base["val_loss"])

    # LoRA B starts at zero, so a fresh adapter scored by the loop's evaluate is the base model
    lora = build_lora_clip(str(tiny_base_dir))
    ds = PairDataset(VAL, caps, paths, cfg.caption_seed)
    ref = evaluate(lora, ds, fake_collate, "cpu", "off", cfg.batch_size)
    assert base == pytest.approx(ref, abs=1e-5)
