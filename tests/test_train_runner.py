from pathlib import Path

import pytest

from pathosearch.eval.golden import write_jsonl
from pathosearch.train.config import TrainConfig
from pathosearch.train.runner import (
    load_manifest,
    run_config,
    run_name,
    select_run_names,
    smoke_config,
)


def make_manifest(d: Path, n_train: int = 50, n_val: int = 10) -> None:
    write_jsonl(
        d / "train_manifest.jsonl",
        [{"image": f"t{i}.jpg", "scale_rank": n_train - 1 - i} for i in range(n_train)],
    )
    write_jsonl(d / "val_manifest.jsonl", [{"image": f"v{i}.jpg"} for i in range(n_val)])
    (d / "train_manifest_meta.json").write_text(
        '{"scales": {"10k": 20, "30k": 40, "full": 50}}', encoding="utf-8"
    )


def test_select_run_names_uses_manifest_scales(tmp_path: Path) -> None:
    make_manifest(tmp_path)
    m = load_manifest(tmp_path)
    train, val = select_run_names(m, TrainConfig(name="x", scale="10k"))
    assert len(train) == 20 and train[0] == "t49.jpg"  # rank 0
    assert len(val) == 10
    train_full, _ = select_run_names(m, TrainConfig(name="x", scale="full"))
    assert train_full[:20] == train and len(train_full) == 50
    with pytest.raises(ValueError, match="scale"):
        select_run_names(m, TrainConfig(name="x", scale="5k"))


def test_smoke_caps_images(tmp_path: Path) -> None:
    make_manifest(tmp_path)
    cfg = smoke_config()
    train, val = select_run_names(load_manifest(tmp_path), cfg)
    assert len(train) == min(cfg.max_train_images or 0, 20) and len(val) == min(
        cfg.max_val_images or 0, 10
    )
    assert cfg.max_steps == 2 and cfg.epochs == 1 and cfg.num_workers == 0


def test_run_config_and_name() -> None:
    assert run_name("30k", 1) == "lora-30k-s1"
    cfg = run_config("full", 0, num_workers=4)
    assert cfg.name == "lora-full-s0" and cfg.scale == "full" and cfg.num_workers == 4
    assert cfg.epochs == 10 and cfg.batch_size == 256  # pre-registered defaults untouched
