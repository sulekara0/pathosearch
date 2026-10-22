"""Training loop on a tiny random CLIP (CPU): run record, checkpoints and exact resume."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import TINY_IMAGE, TINY_VOCAB, make_image_files
from pathosearch.train.config import TrainConfig, config_hash, lr_factor, resolve_amp
from pathosearch.train.data import Batch, Item
from pathosearch.train.loop import retrieval_recall, train
from pathosearch.train.model import build_lora_clip, load_lora_clip

TRAIN = [f"t{i:02d}.jpg" for i in range(12)]
VAL = [f"v{i}.jpg" for i in range(6)]


def fake_collate(items: list[Item]) -> Batch:
    """Deterministic stand-in for CLIPProcessor at the tiny model's sizes."""
    pix = torch.stack(
        [
            torch.from_numpy(
                np.asarray(it.image.resize((TINY_IMAGE, TINY_IMAGE)), dtype=np.float32) / 255.0
            ).permute(2, 0, 1)
            for it in items
        ]
    )
    ids = torch.tensor(
        [
            [0]
            + [2 + (sum(map(ord, it.caption)) * (k + 1)) % (TINY_VOCAB - 2) for k in range(6)]
            + [1]
            for it in items
        ]
    )
    return Batch(
        pixel_values=pix,
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        names=[it.name for it in items],
        captions=[it.caption for it in items],
        truncated=0,
    )


@pytest.fixture
def data(tmp_path: Path):
    paths = make_image_files(tmp_path / "imgs", TRAIN + VAL)
    caps = {n: [f"caption {n}", f"other text {n}"] for n in TRAIN + VAL}
    return paths, caps


def cfg(base: Path, **kw) -> TrainConfig:
    args = dict(
        name="tiny",
        scale="full",
        seed=0,
        base_model=str(base),
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.1,
        lr=5e-3,
        epochs=3,
        batch_size=4,
        num_workers=0,
        amp="auto",
    )
    args.update(kw)
    return TrainConfig(**args)


def run(c: TrainConfig, data, out: Path, **kw):
    paths, caps = data
    return train(
        c,
        TRAIN,
        VAL,
        caps,
        paths,
        out,
        build_model=lambda: build_lora_clip(c.base_model, c.lora_settings()),
        collate=fake_collate,
        device="cpu",
        provenance={"git_sha": "test", "dirty": False},
        log=lambda _: None,
        **kw,
    )


def trainable_state(out: Path) -> dict[str, torch.Tensor]:
    return torch.load(out / "checkpoints" / "last" / "state.pt", weights_only=False)["trainable"]


# --- config helpers ---------------------------------------------------------------------


def test_config_round_trip_and_hash() -> None:
    c = TrainConfig(name="x", scale="10k")
    assert TrainConfig.from_dict(json.loads(json.dumps(c.to_dict()))) == c
    assert config_hash(c) == config_hash(TrainConfig(name="x", scale="10k"))
    assert config_hash(c) != config_hash(TrainConfig(name="x", scale="10k", seed=1))


def test_defaults_match_the_plan() -> None:
    c = TrainConfig(name="x", scale="full")
    assert (c.lora_r, c.lora_alpha, c.lora_dropout, c.lora_targets) == (
        16,
        32,
        0.1,
        ("q_proj", "v_proj"),
    )
    assert (c.lr, c.weight_decay, c.warmup_frac, c.epochs, c.batch_size) == (
        1e-4,
        0.1,
        0.05,
        10,
        256,
    )
    assert (c.grad_clip, c.amp, c.caption_seed) == (1.0, "auto", 42)


def test_resolve_amp() -> None:
    assert resolve_amp("auto", "cpu", bf16_ok=False) == "off"
    assert resolve_amp("auto", "cuda", bf16_ok=True) == "bf16"
    assert resolve_amp("auto", "cuda", bf16_ok=False) == "fp16"
    assert resolve_amp("off", "cuda", bf16_ok=True) == "off"
    with pytest.raises(ValueError):
        resolve_amp("fp16", "cpu", bf16_ok=False)
    with pytest.raises(ValueError):
        resolve_amp("bf16", "cuda", bf16_ok=False)


def test_lr_factor_warmup_then_cosine() -> None:
    total, warm = 100, 10
    assert lr_factor(0, total, warm) == pytest.approx(0.1)
    assert lr_factor(9, total, warm) == pytest.approx(1.0)
    assert lr_factor(10, total, warm) == pytest.approx(1.0)
    assert lr_factor(55, total, warm) == pytest.approx(0.5, abs=1e-6)
    assert lr_factor(99, total, warm) == pytest.approx(0.5 * (1 + math.cos(math.pi * 89 / 90)))
    assert all(lr_factor(s, total, 0) <= 1.0 for s in range(total))


def test_retrieval_recall() -> None:
    img = torch.eye(3)
    txt = torch.tensor([[1.0, 0, 0], [0, 0, 1.0], [0, 1.0, 0]])  # 0 right, 1 and 2 swapped
    r = retrieval_recall(img, txt, ks=(1, 2))
    assert r["i2t_r1"] == pytest.approx(1 / 3) and r["t2i_r1"] == pytest.approx(1 / 3)
    assert r["i2t_r2"] == pytest.approx(1.0)


# --- loop -------------------------------------------------------------------------------


def test_run_record_and_artifacts(tiny_base_dir: Path, data, tmp_path: Path) -> None:
    out = tmp_path / "run"
    rec = run(cfg(tiny_base_dir), data, out)
    on_disk = json.loads((out / "run.json").read_text())
    assert on_disk == json.loads(json.dumps(rec))
    for key in (
        "name", "status", "config", "config_hash", "provenance", "device", "amp", "n_train",
        "n_val", "steps_per_epoch", "total_steps", "params", "history", "best_epoch",
        "best_val_loss", "global_step", "truncated_captions", "resumed_at_epochs", "seconds",
    ):  # fmt: skip
        assert key in rec, key
    assert rec["status"] == "finished" and rec["amp"] == "off"
    assert rec["steps_per_epoch"] == 3 and rec["global_step"] == 9 == rec["total_steps"]
    assert rec["n_train"] == 12 and rec["n_val"] == 6
    assert [h["epoch"] for h in rec["history"]] == [1, 2, 3]
    for h in rec["history"]:
        assert math.isfinite(h["train_loss"]) and math.isfinite(h["val_loss"])
        assert {"i2t_r1", "t2i_r1", "i2t_r5", "t2i_r5", "logit_scale", "lr"} <= set(h)
        assert h["logit_scale"] <= 100.0 + 1e-3
    best = min(rec["history"], key=lambda h: h["val_loss"])
    assert rec["best_epoch"] == best["epoch"] and rec["best_val_loss"] == best["val_loss"]
    assert rec["config_hash"] == config_hash(cfg(tiny_base_dir))
    assert 0 < rec["params"]["trainable"] < rec["params"]["total"]
    model = load_lora_clip(out / "adapter")  # best checkpoint is a loadable adapter
    assert model.lora_meta["epoch"] == rec["best_epoch"]


def test_training_moves_the_model(tiny_base_dir: Path, data, tmp_path: Path) -> None:
    rec = run(cfg(tiny_base_dir, epochs=4, lora_dropout=0.0), data, tmp_path / "run")
    assert rec["history"][-1]["train_loss"] < rec["history"][0]["train_loss"]
    assert rec["history"][-1]["logit_scale"] != pytest.approx(math.exp(2.5), abs=1e-6)  # learned


def test_max_steps_stops_early(tiny_base_dir: Path, data, tmp_path: Path) -> None:
    rec = run(cfg(tiny_base_dir, max_steps=2), data, tmp_path / "run")
    assert rec["global_step"] == 2 and rec["status"] == "finished"
    assert len(rec["history"]) == 1  # the partial epoch is still validated and saved


def test_resume_is_exact(tiny_base_dir: Path, data, tmp_path: Path) -> None:
    c = cfg(tiny_base_dir)
    full = run(c, data, tmp_path / "full")

    part_dir = tmp_path / "part"
    first = run(c, data, part_dir, stop_after_epoch=1)
    assert first["status"] == "stopped" and first["global_step"] == 3
    resumed = run(c, data, part_dir, resume=True)

    assert resumed["status"] == "finished" and resumed["resumed_at_epochs"] == [1]
    assert resumed["global_step"] == full["global_step"]
    assert [h["train_loss"] for h in resumed["history"]] == [
        h["train_loss"] for h in full["history"]
    ]
    a, b = trainable_state(tmp_path / "full"), trainable_state(part_dir)
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_refuses_to_overwrite_or_resume_with_other_config(
    tiny_base_dir: Path, data, tmp_path: Path
) -> None:
    out = tmp_path / "run"
    run(cfg(tiny_base_dir), data, out, stop_after_epoch=1)
    with pytest.raises(FileExistsError):
        run(cfg(tiny_base_dir), data, out)
    with pytest.raises(ValueError, match="config"):
        run(cfg(tiny_base_dir, lr=1e-3), data, out, resume=True)


def test_resume_without_checkpoint_fails(tiny_base_dir: Path, data, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run(cfg(tiny_base_dir), data, tmp_path / "none", resume=True)


def test_val_and_train_must_be_disjoint(tiny_base_dir: Path, data, tmp_path: Path) -> None:
    paths, caps = data
    with pytest.raises(ValueError, match="overlap"):
        train(
            cfg(tiny_base_dir), TRAIN, VAL + [TRAIN[0]], caps, paths, tmp_path / "r",
            build_model=lambda: None, collate=fake_collate, device="cpu",
        )  # fmt: skip
