"""run_eval additions for R4: adapter systems, explicit comparisons, judged@k."""

import json
from pathlib import Path

import pytest

from pathosearch.eval.run_eval import (
    adapter_summary,
    encoder_arg,
    judged_at_k,
    parse_system,
    render_markdown,
    run_eval,
)
from test_run_eval import make_golden, runners


def test_parse_system_accepts_registered_adapters() -> None:
    sp = parse_system("lora-full-s0", extra_encoders={"lora-full-s0"})
    assert sp.encoder == "lora-full-s0" and not sp.rerank
    assert parse_system("lora-full-s0+rerank", extra_encoders={"lora-full-s0"}).rerank
    with pytest.raises(ValueError):
        parse_system("lora-full-s0")
    with pytest.raises(ValueError):  # adapter names must not shadow built-in encoders
        parse_system("clip", extra_encoders={"clip"})


def test_encoder_arg() -> None:
    adapters = {"lora-10k-s0": Path("runs/10k/adapter")}
    assert encoder_arg("clip", adapters) == "clip"
    assert encoder_arg("lora-10k-s0", adapters) == "lora:runs/10k/adapter"


def test_judged_at_k() -> None:
    runs = {"a": {"q1": ["x", "y", "z"], "q2": ["x"]}}
    judged = {"q1": {"x", "z"}, "q2": set()}
    out = judged_at_k(runs, judged, k=2)
    assert out["a"]["per_query"] == {"q1": 0.5, "q2": 0.0}
    assert out["a"]["mean"] == pytest.approx(0.25)


def test_run_eval_reports_judged_at_k_including_unsure(tmp_path: Path) -> None:
    g = make_golden(tmp_path)
    report = run_eval(
        g, "dev", ["clip", "quilt"], 3, runners(), {}, "c", n_boot=50, sha="x", packages={}
    )
    jk = report["judged_at_k"]
    # clip (oracle): q1 3/3; q2 img3,img4 judged, img0 not; q3 only img5 judged
    assert jk["clip"]["per_query"] == pytest.approx({"q1": 1.0, "q2": 2 / 3, "q3": 1 / 3}, abs=1e-4)
    # quilt: q2 ranks img7, which A marked unsure: still counts as judged (a human saw it)
    assert jk["quilt"]["per_query"]["q2"] == pytest.approx(1.0)
    assert "judged@3" in render_markdown(report)


def test_explicit_comparisons(tmp_path: Path) -> None:
    g = make_golden(tmp_path)
    report = run_eval(
        g, "dev", ["clip", "quilt"], 3, runners(), {}, "c", n_boot=50, sha="x", packages={},
        comparisons=[("clip", "quilt")],
    )  # fmt: skip
    comps = report["metrics"]["lenient"]["comparisons"]["ndcg"]
    assert [(c["system"], c["baseline"]) for c in comps] == [("clip", "quilt")]
    assert report["comparisons"] == [["clip", "quilt"]]
    with pytest.raises(ValueError, match="unknown system"):
        run_eval(
            g, "dev", ["clip", "quilt"], 3, runners(), {}, "c", n_boot=50, sha="x", packages={},
            comparisons=[("clip", "lora")],
        )  # fmt: skip


def test_adapters_are_recorded_in_config(tmp_path: Path) -> None:
    g = make_golden(tmp_path)
    info = {"lora-x": {"dir": "d", "epoch": 2}}
    report = run_eval(
        g, "dev", ["clip", "quilt"], 3, runners(), {}, "c", n_boot=50, sha="x", packages={},
        adapters=info,
    )  # fmt: skip
    assert report["config"]["adapters"] == info


def test_adapter_summary_reads_meta_and_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "adapter").mkdir(parents=True)
    (run_dir / "adapter" / "lora_meta.json").write_text(
        json.dumps(
            {"base_model": "b", "epoch": 4, "config_hash": "h", "log_scale": 4.0, "lora": {"r": 16}}
        )
    )
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "provenance": {"git_sha": "abc", "dirty": False},
                "best_val_loss": 1.5,
                "status": "finished",
            }
        )
    )
    s = adapter_summary(run_dir / "adapter")
    assert s == {
        "dir": (run_dir / "adapter").as_posix(),
        "base_model": "b",
        "epoch": 4,
        "config_hash": "h",
        "lora": {"r": 16},
        "train_git_sha": "abc",
        "train_dirty": False,
        "best_val_loss": 1.5,
        "train_status": "finished",
    }
    (run_dir / "run.json").unlink()
    assert adapter_summary(run_dir / "adapter")["train_git_sha"] is None
