"""Symmetric InfoNCE: checked against an independent numpy reference."""

import math

import numpy as np
import pytest
import torch

from pathosearch.train.loss import (
    MAX_LOGIT_SCALE,
    LearnableTemperature,
    clip_infonce,
    clip_infonce_terms,
    same_pair_mask,
)


def ref_loss(img: np.ndarray, txt: np.ndarray, scale: float, same: np.ndarray | None) -> float:
    """Plain loops: for each anchor, -log softmax over its own positive and allowed negatives."""
    i = img / np.linalg.norm(img, axis=1, keepdims=True)
    t = txt / np.linalg.norm(txt, axis=1, keepdims=True)
    s = scale * (i @ t.T)
    n = len(s)
    allowed = np.ones((n, n), bool) if same is None else ~same | np.eye(n, dtype=bool)

    def ce(m: np.ndarray, ok: np.ndarray) -> float:
        tot = 0.0
        for a in range(n):
            row = m[a][ok[a]]
            tot += -(m[a, a] - (np.log(np.exp(row - row.max()).sum()) + row.max()))
        return tot / n

    return (ce(s, allowed) + ce(s.T, allowed.T)) / 2


def rand(n: int = 6, d: int = 8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)).astype(np.float32), rng.normal(size=(n, d)).astype(np.float32)


def t(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x)


def test_matches_reference_without_mask() -> None:
    img, txt = rand()
    got = clip_infonce(t(img), t(txt), torch.tensor(7.5))
    assert got.item() == pytest.approx(ref_loss(img, txt, 7.5, None), rel=1e-5)


def test_inputs_are_normalised_inside() -> None:
    img, txt = rand()
    a = clip_infonce(t(img), t(txt), torch.tensor(10.0))
    b = clip_infonce(t(img * 3.0), t(txt * 0.1), torch.tensor(10.0))
    assert a.item() == pytest.approx(b.item(), rel=1e-5)


def test_masked_pairs_are_not_negatives() -> None:
    img, txt = rand(n=5)
    same = np.zeros((5, 5), bool)
    same[0, 3] = same[3, 0] = True  # e.g. two captions of one image / identical captions
    got = clip_infonce(t(img), t(txt), torch.tensor(5.0), torch.from_numpy(same))
    assert got.item() == pytest.approx(ref_loss(img, txt, 5.0, same), rel=1e-5)
    assert got.item() != pytest.approx(ref_loss(img, txt, 5.0, None), rel=1e-3)


def test_masked_negative_has_no_influence() -> None:
    """Moving a masked text anywhere must not change anchor 0's image->text term."""
    img, txt = rand(n=4)
    same = torch.zeros(4, 4, dtype=torch.bool)
    same[0, 2] = same[2, 0] = True
    base = clip_infonce_terms(t(img), t(txt), torch.tensor(5.0), same)
    txt2 = txt.copy()
    txt2[2] = img[0]  # masked text 2 now matches image 0 perfectly
    moved = clip_infonce_terms(t(img), t(txt2), torch.tensor(5.0), same)
    assert moved["i2t"][0].item() == pytest.approx(base["i2t"][0].item(), rel=1e-6)


def test_diagonal_of_mask_is_ignored() -> None:
    img, txt = rand()
    eye = torch.eye(6, dtype=torch.bool)
    a = clip_infonce(t(img), t(txt), torch.tensor(5.0), eye)
    b = clip_infonce(t(img), t(txt), torch.tensor(5.0))
    assert a.item() == pytest.approx(b.item(), rel=1e-6)


def test_fully_masked_rows_give_zero_loss() -> None:
    img, txt = rand(n=3)
    same = torch.ones(3, 3, dtype=torch.bool)
    assert clip_infonce(t(img), t(txt), torch.tensor(5.0), same).item() == pytest.approx(0.0)


def test_aligned_pairs_beat_shuffled() -> None:
    img, _ = rand(n=8)
    aligned = clip_infonce(t(img), t(img), torch.tensor(50.0)).item()
    shuffled = clip_infonce(t(img), t(img[::-1].copy()), torch.tensor(50.0)).item()
    assert aligned < 0.05 < shuffled


def test_scale_is_used_as_is_and_clamping_belongs_to_temperature() -> None:
    img, txt = rand()
    a = clip_infonce(t(img), t(txt), torch.tensor(200.0)).item()
    assert a == pytest.approx(ref_loss(img, txt, 200.0, None), rel=1e-5)


def test_gradients_reach_inputs_and_temperature() -> None:
    img, txt = rand()
    ti, tt = t(img).requires_grad_(), t(txt).requires_grad_()
    temp = LearnableTemperature(init=math.log(20.0))
    clip_infonce(ti, tt, temp()).backward()
    assert ti.grad is not None and tt.grad is not None
    assert temp.log_scale.grad is not None and temp.log_scale.grad.item() != 0.0


def test_temperature_init_and_clamp() -> None:
    temp = LearnableTemperature()  # CLIP's pretrained value ln(100)
    assert temp.log_scale.requires_grad
    assert temp().item() == pytest.approx(100.0, rel=1e-5)
    # an init above the cap is clamped at construction
    assert LearnableTemperature(init=math.log(1000.0))().item() == pytest.approx(MAX_LOGIT_SCALE)
    # after an optimizer step pushes it above the cap, clamp_() pulls it back
    with torch.no_grad():
        temp.log_scale.fill_(math.log(1000.0))
    temp.clamp_()
    assert temp().item() == pytest.approx(MAX_LOGIT_SCALE)
    assert temp.log_scale.item() <= math.log(MAX_LOGIT_SCALE) + 1e-6  # float32 rounding


def test_temperature_learns_at_the_cap() -> None:
    """Regression: CLIP's stored logit_scale is a hair above ln(100) (exp = 100.0000064).

    Clamping inside forward() zeroed the gradient there, freezing the temperature; the cap
    is applied to the parameter instead, so the gradient must be non-zero at init.
    """
    temp = LearnableTemperature(init=4.605170249938965)
    temp().backward()
    assert temp.log_scale.grad is not None and temp.log_scale.grad.item() != 0.0
    img, txt = rand()
    temp.log_scale.grad = None
    clip_infonce(t(img), t(txt), temp()).backward()
    assert temp.log_scale.grad is not None and temp.log_scale.grad.item() != 0.0


def test_half_precision_inputs_compute_in_float32() -> None:
    img, txt = rand()
    out = clip_infonce(t(img).half(), t(txt).half(), torch.tensor(100.0))
    assert out.dtype == torch.float32 and torch.isfinite(out)
    assert out.item() == pytest.approx(ref_loss(img, txt, 100.0, None), rel=2e-2)


def test_shape_checks() -> None:
    img, txt = rand()
    with pytest.raises(ValueError, match="same shape"):
        clip_infonce(t(img), t(txt[:5]), torch.tensor(1.0))
    with pytest.raises(ValueError, match="mask"):
        clip_infonce(t(img), t(txt), torch.tensor(1.0), torch.zeros(5, 5, dtype=torch.bool))


def test_same_pair_mask_from_ids_and_captions() -> None:
    m = same_pair_mask(["a", "b", "a", "c"], ["x", "Same  Cap", "y", " same cap"])
    expected = torch.tensor(
        [
            [True, False, True, False],  # a == a (same image)
            [False, True, False, True],  # b ~ c (equal normalised caption)
            [True, False, True, False],
            [False, True, False, True],
        ]
    )
    assert torch.equal(m, expected)
    with pytest.raises(ValueError):
        same_pair_mask(["a"], ["x", "y"])
