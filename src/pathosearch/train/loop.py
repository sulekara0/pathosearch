"""LoRA training loop: AdamW + warmup/cosine, optional AMP, per-epoch validation,
best-adapter export, and an end-of-epoch checkpoint that resumes bit-for-bit (on CPU).

Layout of ``out_dir``::

    run.json                  run record (rewritten after every epoch)
    adapter/                  best epoch by internal validation loss (pre-registered rule)
    checkpoints/last/state.pt trainable weights, optimizer, scheduler, scaler, RNG, history

Resume restores everything at an epoch boundary; the epoch's batch order is a function of
(seed, epoch), so the resumed run sees the same batches as an uninterrupted one. The golden
set is never touched here: validation uses the internal split only.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import random
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from pathosearch.train.config import TrainConfig, config_hash, lr_factor, resolve_amp
from pathosearch.train.data import Batch, Item, PairDataset, epoch_order
from pathosearch.train.loss import clip_infonce, same_pair_mask
from pathosearch.train.model import LoraCLIP, count_parameters, save_lora_clip

Collate = Callable[[list[Item]], Batch]
Log = Callable[[str], None]
RECALL_KS = (1, 5)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _set_rng_state(s: Mapping[str, Any]) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def retrieval_recall(
    img: torch.Tensor, txt: torch.Tensor, ks: Sequence[int] = RECALL_KS
) -> dict[str, float]:
    """Recall@k for paired rows, both directions (ties count in favour of the pair)."""
    sims = (
        torch.nn.functional.normalize(img.float(), dim=-1)
        @ torch.nn.functional.normalize(txt.float(), dim=-1).T
    )
    diag = sims.diagonal()
    rank_i2t = (sims > diag[:, None]).sum(dim=1)  # 0 = correct text ranked first
    rank_t2i = (sims > diag[None, :]).sum(dim=0)
    out: dict[str, float] = {}
    for k in ks:
        out[f"i2t_r{k}"] = float((rank_i2t < k).float().mean())
        out[f"t2i_r{k}"] = float((rank_t2i < k).float().mean())
    return out


def _autocast(device: str, amp: str) -> contextlib.AbstractContextManager[Any]:
    if amp == "off":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _encode(model: LoraCLIP, b: Batch, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    img = model.encode_image(b.pixel_values.to(device))
    txt = model.encode_text(b.input_ids.to(device), b.attention_mask.to(device))
    return img, txt


@torch.no_grad()
def evaluate(
    model: LoraCLIP, ds: PairDataset, collate: Collate, device: str, amp: str, batch_size: int
) -> dict[str, float]:
    """Loss over the whole validation set at once (independent of batch size) and recall."""
    model.eval()
    ds.set_epoch(0)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    imgs, txts, names, caps = [], [], [], []
    for b in loader:
        with _autocast(device, amp):
            i, t = _encode(model, b, device)
        imgs.append(i.float().cpu())
        txts.append(t.float().cpu())
        names += b.names
        caps += b.captions
    img, txt = torch.cat(imgs), torch.cat(txts)
    scale = model.temperature().detach().cpu()
    loss = clip_infonce(img, txt, scale, same_pair_mask(names, caps))
    return {"val_loss": float(loss), **retrieval_recall(img, txt)}


def _write_json_atomic(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _save_adapter_atomic(model: LoraCLIP, out: Path, extra: Mapping[str, Any]) -> None:
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    save_lora_clip(model, tmp, extra)
    if out.exists():
        shutil.rmtree(out)
    os.replace(tmp, out)


def _trainable(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def train(
    cfg: TrainConfig,
    train_names: Sequence[str],
    val_names: Sequence[str],
    captions: Mapping[str, Sequence[str]],
    paths: Mapping[str, Path],
    out_dir: Path,
    build_model: Callable[[], LoraCLIP | None],
    collate: Collate,
    device: str = "cpu",
    provenance: Mapping[str, Any] | None = None,
    resume: bool = False,
    stop_after_epoch: int | None = None,
    log: Log = print,
) -> dict[str, Any]:
    """Train and return the run record (also written to ``out_dir/run.json``).

    ``stop_after_epoch`` ends the session after that epoch with status ``stopped`` (used to
    split a run over sessions, and in tests to simulate an interruption).
    """
    overlap = set(train_names) & set(val_names)
    if overlap:
        raise ValueError(f"train and val overlap in {len(overlap)} images")
    state_path = out_dir / "checkpoints" / "last" / "state.pt"
    if resume and not state_path.is_file():
        raise FileNotFoundError(f"nothing to resume: {state_path}")
    if not resume and out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty; pass resume=True or use a new directory")

    bf16_ok = device.startswith("cuda") and torch.cuda.is_bf16_supported()
    amp = resolve_amp(cfg.amp, device, bf16_ok)
    seed_everything(cfg.seed)
    model = build_model()
    if model is None:
        raise ValueError("build_model returned None")
    model.to(device)

    params = _trainable(model)
    temp = model.temperature.log_scale
    lora = [p for p in params.values() if p is not temp]
    opt = torch.optim.AdamW(
        [
            {"params": lora, "weight_decay": cfg.weight_decay},
            {"params": [temp], "weight_decay": 0.0},
        ],
        lr=cfg.lr,
    )
    train_ds = PairDataset(train_names, captions, paths, cfg.caption_seed)
    val_ds = PairDataset(val_names, captions, paths, cfg.caption_seed)
    steps_per_epoch = len(train_ds) // cfg.batch_size
    if steps_per_epoch == 0:
        raise ValueError(f"{len(train_ds)} training images < batch size {cfg.batch_size}")
    total_steps = steps_per_epoch * cfg.epochs
    if cfg.max_steps is not None:
        total_steps = min(total_steps, cfg.max_steps)
    warmup = round(cfg.warmup_frac * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, total_steps, warmup))
    scaler = torch.amp.GradScaler("cuda", enabled=amp == "fp16")

    h = config_hash(cfg)
    start_epoch, global_step = 0, 0
    history: list[dict[str, Any]] = []
    best_epoch: int | None = None
    best_loss: float | None = None
    truncated = 0
    seconds = 0.0
    resumed: list[int] = []
    if resume:
        st = torch.load(state_path, weights_only=False, map_location="cpu")
        if st["config_hash"] != h:
            raise ValueError("checkpoint was written with a different config; refusing to resume")
        if set(st["trainable"]) != set(params):
            raise ValueError("checkpoint trainable parameters do not match the model")
        with torch.no_grad():
            for n, p in params.items():
                p.copy_(st["trainable"][n].to(device))
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        _set_rng_state(st["rng"])
        start_epoch, global_step = st["epoch"], st["global_step"]
        history, best_epoch, best_loss = st["history"], st["best_epoch"], st["best_val_loss"]
        truncated, seconds = st["truncated"], st["seconds"]
        resumed = [*st["resumed_at_epochs"], start_epoch]
        log(f"resumed at epoch {start_epoch}, step {global_step}")

    record: dict[str, Any] = {
        "name": cfg.name,
        "status": "running",
        "config": cfg.to_dict(),
        "config_hash": h,
        "provenance": dict(provenance or {}),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "amp": amp,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "warmup_steps": warmup,
        "params": count_parameters(model),
    }

    def finish(status: str) -> dict[str, Any]:
        record.update(
            status=status,
            history=history,
            best_epoch=best_epoch,
            best_val_loss=best_loss,
            global_step=global_step,
            truncated_captions=truncated,
            resumed_at_epochs=resumed,
            seconds=round(seconds, 2),
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(out_dir / "run.json", record)
        return record

    for epoch in range(start_epoch, cfg.epochs):
        if global_step >= total_steps:
            break
        t0 = time.perf_counter()
        train_ds.set_epoch(epoch)
        loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            sampler=epoch_order(len(train_ds), cfg.seed, epoch),
            drop_last=True,
            num_workers=cfg.num_workers,
            collate_fn=collate,
            pin_memory=device.startswith("cuda"),
        )
        model.train()
        losses: list[float] = []
        for b in loader:
            if global_step >= total_steps:
                break
            with _autocast(device, amp):
                img, txt = _encode(model, b, device)
            loss = clip_infonce(img, txt, model.temperature(), same_pair_mask(b.names, b.captions))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            clip_grad_norm_(list(params.values()), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            model.temperature.clamp_()  # cap on the parameter keeps its gradient alive
            sched.step()
            global_step += 1
            truncated += b.truncated
            losses.append(float(loss.detach()))
            if not math.isfinite(losses[-1]):
                raise FloatingPointError(f"non-finite loss at step {global_step}")

        val = evaluate(model, val_ds, collate, device, amp, cfg.batch_size)
        seconds += time.perf_counter() - t0
        history.append(
            {
                "epoch": epoch + 1,
                "steps": len(losses),
                "train_loss": float(np.mean(losses)),
                **val,
                "logit_scale": float(model.temperature().detach()),
                "lr": float(sched.get_last_lr()[0]),
                "seconds": round(time.perf_counter() - t0, 2),
            }
        )
        log(json.dumps(history[-1]))
        if best_loss is None or val["val_loss"] < best_loss:
            best_epoch, best_loss = epoch + 1, val["val_loss"]
            _save_adapter_atomic(
                model, out_dir / "adapter", {"run": cfg.name, "epoch": best_epoch, "config_hash": h}
            )
        state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = state_path.with_name("state.pt.tmp")
        torch.save(
            {
                "config_hash": h,
                "epoch": epoch + 1,
                "global_step": global_step,
                "trainable": {n: p.detach().cpu().clone() for n, p in params.items()},
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "scaler": scaler.state_dict(),
                "rng": _rng_state(),
                "history": history,
                "best_epoch": best_epoch,
                "best_val_loss": best_loss,
                "truncated": truncated,
                "seconds": seconds,
                "resumed_at_epochs": resumed,
            },
            tmp,
        )
        os.replace(tmp, state_path)
        if (
            stop_after_epoch is not None
            and epoch + 1 >= stop_after_epoch
            and epoch + 1 < cfg.epochs
        ):
            return finish("stopped")
        finish("running")
    return finish("finished")
