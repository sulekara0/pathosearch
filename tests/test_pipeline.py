import faiss
import numpy as np
import pytest

from pathosearch.faiss_search import raw_search
from pathosearch.pipeline import SearchIndex, run_text_search, search_pipeline

DIM = 4
THRESHOLD = 0.80


class FakeGate:
    def __init__(self, score: float) -> None:
        self._score = score

    def score(self, query: str) -> float:
        return self._score


class FakeEncoder:
    """Returns a fixed unit vector for any text."""

    def __init__(self, vec: list[float]) -> None:
        self._vec = np.array([vec], dtype=np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return self._vec


def _unit(v: list[float]) -> list[float]:
    a = np.array(v, dtype=np.float32)
    return (a / np.linalg.norm(a)).tolist()


def make_index(
    vectors: list[list[float]], ids: list[str], captions: dict[str, dict]
) -> SearchIndex:
    embs = np.array([_unit(v) for v in vectors], dtype=np.float32)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return SearchIndex(index, "ip", ids, captions)


@pytest.fixture
def idx() -> SearchIndex:
    # cosine to query [1,0,0,0] decreases from a.jpg to d.jpg
    vectors = [[1, 0, 0, 0], [1, 0.05, 0, 0], [1, 0.1, 0, 0], [1, 1, 0, 0]]
    ids = ["sub/a.jpg", "sub/b.jpg", "sub/c.jpg", "sub/d.jpg"]
    captions = {
        "a.jpg": {"caption": "skin biopsy", "pathology": "Dermatopathology"},
        "b.jpg": {"caption": "kidney tissue", "pathology": "Renal"},
        "c.jpg": {"caption": "lung adenocarcinoma biopsy", "pathology": "Pulmonary"},
        "d.jpg": {"caption": "lung adenocarcinoma lung cancer", "pathology": "Pulmonary"},
    }
    return make_index(vectors, ids, captions)


def _loader(idx: SearchIndex):
    enc = FakeEncoder([1, 0, 0, 0])
    return lambda: (enc, idx)


def test_out_of_domain_exits_early_without_touching_retrieval() -> None:
    def must_not_be_called():
        raise AssertionError("retrieval must not run for out-of-domain queries")

    res = search_pipeline("araba motoru", FakeGate(0.5), must_not_be_called, 4, True, THRESHOLD)

    assert res["status"] == "domain_out"
    assert res["results"] == []
    assert res["faiss_called"] is False
    assert res["reranked"] is False
    assert res["domain_score"] == 0.5


def test_in_domain_without_rerank_returns_faiss_order(idx: SearchIndex) -> None:
    res = search_pipeline("lung", FakeGate(0.9), _loader(idx), 3, False, THRESHOLD)

    assert res["status"] == "ok"
    assert res["faiss_called"] is True
    assert res["reranked"] is False
    assert [r["id"] for r in res["results"]] == ["sub/a.jpg", "sub/b.jpg", "sub/c.jpg"]
    assert res["results"][0]["caption"] == "skin biopsy"  # basename caption lookup
    assert res["enc_ms"] >= 0 and res["srch_ms"] >= 0


def test_rerank_fetches_larger_pool_and_uses_caption_overlap(idx: SearchIndex) -> None:
    # topk=1: without rerank the winner is a.jpg (highest cosine). With rerank the pool is
    # topk*5 (all 4 docs) and the lung captions overtake the skin caption.
    plain = search_pipeline("lung adenocarcinoma", FakeGate(0.9), _loader(idx), 1, False, THRESHOLD)
    reranked = search_pipeline(
        "lung adenocarcinoma", FakeGate(0.9), _loader(idx), 1, True, THRESHOLD
    )

    assert plain["results"][0]["id"] == "sub/a.jpg"
    assert reranked["reranked"] is True
    assert reranked["results"][0]["pathology"] == "Pulmonary"
    assert "rerank_score" in reranked["results"][0]


def test_threshold_boundary_is_in_domain(idx: SearchIndex) -> None:
    res = search_pipeline("q", FakeGate(THRESHOLD), _loader(idx), 2, False, THRESHOLD)
    assert res["status"] == "ok"


def test_run_text_search_reports_timings(idx: SearchIndex) -> None:
    results, enc_ms, srch_ms = run_text_search(FakeEncoder([1, 0, 0, 0]), idx, "q", 2)
    assert len(results) == 2
    assert enc_ms >= 0 and srch_ms >= 0


def test_raw_search_l2_negates_distance_and_falls_back_to_full_id() -> None:
    embs = np.array([[0, 0, 0, 0], [3, 0, 0, 0]], dtype=np.float32)
    index = faiss.IndexFlatL2(DIM)
    index.add(embs)
    caps = {"x/full_id.jpg": {"caption": "c", "pathology": "p"}}

    out = raw_search(index, "l2", ["x/full_id.jpg", "other.jpg"], caps, embs[:1], 2)

    assert out[0]["id"] == "x/full_id.jpg"
    assert out[0]["score"] == 0.0 and out[1]["score"] == -9.0
    assert out[0]["caption"] == "c" and out[1]["caption"] == ""
