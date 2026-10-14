import json
from pathlib import Path

from pathosearch.eval.metrics import is_relevant, precision_at_k
from pathosearch.faiss_search import load_captions
from pathosearch.labels import normalize_labels, normalize_pathology
from pathosearch.reranker import rerank_results


def test_normalize_pathology_case_and_whitespace() -> None:
    assert normalize_pathology("Unknown") == normalize_pathology("unknown") == "unknown"
    assert normalize_pathology("  Soft tissue \n") == "soft tissue"
    assert normalize_pathology(None) == ""
    assert normalize_labels(["Renal", " renal", "Bone"]) == {"renal", "bone"}


def test_load_captions_normalizes_pathology(tmp_path: Path) -> None:
    f = tmp_path / "c.jsonl"
    rows = [
        {"path": "a.jpg", "caption": "x", "pathology": "Unknown"},
        {"path": "b.jpg", "caption": "y", "pathology": " unknown "},
        {"path": "c.jpg", "caption": "z", "pathology": "Dermatopathology"},
    ]
    f.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    caps = load_captions(f)
    assert caps["a.jpg"]["pathology"] == caps["b.jpg"]["pathology"] == "unknown"
    assert caps["c.jpg"]["pathology"] == "dermatopathology"


def test_is_relevant_ignores_case_on_both_sides() -> None:
    assert is_relevant({"pathology": "dermatopathology"}, ["Dermatopathology"])
    assert is_relevant({"pathology": "Renal "}, ["renal"])
    assert not is_relevant({"pathology": "renal"}, [])
    assert precision_at_k([{"pathology": "renal"}], ["Renal"], 1) == 1.0


def test_rerank_pathology_bonus_ignores_case() -> None:
    cands = [
        {"id": "a", "score": 0.3, "caption": "", "pathology": "renal"},
        {"id": "b", "score": 0.3, "caption": "", "pathology": "bone"},
    ]
    out = rerank_results("q", cands, expected_categories=["Bone"], gamma=1.0, topk=2)
    assert out[0]["id"] == "b"


def test_parse_pathology_and_balanced_targets_are_normalized() -> None:
    from pathosearch.data.prepare_quilt import BALANCED_TARGETS, parse_pathology

    assert parse_pathology("['Dermatopathology', 'Bone']") == "dermatopathology"
    assert parse_pathology("Unknown") == "unknown"
    assert parse_pathology("") == "unknown"
    assert all(k == normalize_pathology(k) for k in BALANCED_TARGETS)
