"""CLI wrappers run end to end on a fake golden directory (no models, no real data)."""

import json
import subprocess
import sys
from pathlib import Path

from pathosearch.eval.fake_golden import make_fake_golden
from pathosearch.eval.golden import (
    GoldenPaths,
    Judgment,
    append_judgment,
    read_jsonl,
    read_trec_qrels,
)

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args], capture_output=True, text=True
    )


def test_make_splits_writes_once_and_refuses_overwrite(tmp_path: Path) -> None:
    fake = make_fake_golden(tmp_path, n_images=30, n_queries=4, depth=3, n_random=1)
    done = run("make_splits.py", "--golden", str(fake.golden_dir), "--force_dev", "q001")
    assert done.returncode == 0, done.stderr
    splits = json.loads(GoldenPaths(fake.golden_dir).splits.read_text(encoding="utf-8"))
    assert "q001" in splits["dev"] and len(splits["dev"]) + len(splits["test"]) == 4
    again = run("make_splits.py", "--golden", str(fake.golden_dir))
    assert again.returncode != 0 and "must not change" in again.stderr


def test_agreement_report_and_qrels_gate(tmp_path: Path) -> None:
    fake = make_fake_golden(tmp_path, n_images=30, n_queries=3, depth=3, n_random=1)
    g = GoldenPaths(fake.golden_dir)
    # kappa is computed on exactly the overlap pairs, so both sides judge all of them
    pairs = sorted((r["qid"], r["image"]) for r in read_jsonl(g.overlap))
    assert len(pairs) == 3
    for (qid, image), (a, b) in zip(pairs, [(2, 2), (0, 0), (2, 0)], strict=True):
        append_judgment(g.judgments("A"), Judgment(qid, image, a, "A", False, "t"))
        append_judgment(g.judgments("B"), Judgment(qid, image, b, "B", False, "t"))
    out = tmp_path / "reports"

    res = run(
        "agreement_report.py", "--golden", str(g.root), "--out-dir", str(out), "--write_qrels"
    )
    assert res.returncode != 0 and "not adjudicated" in res.stderr  # one open disagreement
    report = json.loads((out / f"agreement_{g.root.name}.json").read_text(encoding="utf-8"))
    assert report["pairs"] == 3 and report["unresolved_disagreements"] == 1
    assert (out / f"agreement_{g.root.name}.md").is_file()
    assert not g.qrels.exists()

    qid, image = pairs[2]
    append_judgment(g.adjudication, Judgment(qid, image, 1, "adjudicated", False, "t"))
    res = run(
        "agreement_report.py", "--golden", str(g.root), "--out-dir", str(out), "--write_qrels"
    )
    assert res.returncode == 0, res.stderr
    assert read_trec_qrels(g.qrels)[qid][image] == 1  # adjudication overrides A
