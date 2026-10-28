"""API tests with a fake gate/encoder and a tiny FAISS index (no models, no real data)."""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import faiss
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pathosearch.api import app as api_app
from pathosearch.api.app import create_app
from pathosearch.api.schemas import DISCLAIMER, GATE_CALIBRATION_NOTE
from pathosearch.api.service import SearchService, load_service
from pathosearch.config import load_settings
from pathosearch.errors import DataNotFoundError
from pathosearch.pipeline import SearchIndex

THRESHOLD = 0.80


class FakeGate:
    def __init__(self, score: float) -> None:
        self.value = score

    def score(self, query: str) -> float:
        return self.value


class FakeEncoder:
    """Same unit vector for every text/image; counts calls."""

    def __init__(self) -> None:
        self.calls = 0
        self._vec = np.array([[1, 0, 0, 0]], dtype=np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        return self._vec

    def encode_images(self, images: list) -> np.ndarray:
        self.calls += 1
        return self._vec


def _index() -> SearchIndex:
    # cosine to [1,0,0,0] decreases from a.jpg to d.jpg
    vectors = np.array(
        [[1, 0, 0, 0], [1, 0.05, 0, 0], [1, 0.1, 0, 0], [1, 1, 0, 0]], dtype=np.float32
    )
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(4)
    index.add(vectors)
    captions = {
        "a.jpg": {"caption": "skin biopsy", "pathology": "dermatopathology"},
        "b.jpg": {"caption": "kidney tissue", "pathology": "renal"},
        "c.jpg": {"caption": "lung adenocarcinoma biopsy", "pathology": "pulmonary"},
        "d.jpg": {"caption": "lung adenocarcinoma", "pathology": "pulmonary"},
    }
    return SearchIndex(index, "ip", ["a.jpg", "b.jpg", "c.jpg", "d.jpg"], captions)


@pytest.fixture
def gate() -> FakeGate:
    return FakeGate(0.9)


@pytest.fixture
def encoder() -> FakeEncoder:
    return FakeEncoder()


@pytest.fixture
def client(gate: FakeGate, encoder: FakeEncoder) -> Iterator[TestClient]:
    svc = SearchService(
        gate=gate,
        encoder=encoder,
        index=_index(),
        gate_threshold=THRESHOLD,
        startup_ms={"total": 1.0},
    )
    with TestClient(create_app(svc)) as c:
        yield c


def _png(size: tuple[int, int] = (8, 8)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 100, 150)).save(buf, format="PNG")
    return buf.getvalue()


# --- /health -----------------------------------------------------------------


def test_health_reports_model_size_and_uncalibrated_gate(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "hf-hub:wisdomik/QuiltNet-B-32"
    assert body["n_vectors"] == 4
    assert body["gate"] == {
        "threshold": THRESHOLD,
        "calibrated": False,
        "note": GATE_CALIBRATION_NOTE,
    }
    assert body["startup_ms"] == {"total": 1.0}


# --- /search/text ------------------------------------------------------------


def test_text_search_in_domain_returns_cases_by_cosine(client: TestClient) -> None:
    r = client.post("/search/text", json={"query": "lung adenocarcinoma", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["kind"] == "text"
    assert [c["case_id"] for c in body["results"]] == ["a.jpg", "b.jpg", "c.jpg"]
    assert body["results"][2]["caption"] == "lung adenocarcinoma biopsy"
    assert body["gate"] == {
        "status": "in_domain",
        "score": 0.9,
        "threshold": THRESHOLD,
        "calibrated": False,
    }
    assert body["reranked"] is False
    assert body["disclaimer"] == DISCLAIMER
    assert set(body["timing"]) == {"encode_ms", "search_ms", "total_ms"}


def test_text_search_rerank_is_opt_in(client: TestClient) -> None:
    plain = client.post("/search/text", json={"query": "lung adenocarcinoma", "top_k": 2}).json()
    reranked = client.post(
        "/search/text", json={"query": "lung adenocarcinoma", "top_k": 2, "rerank": True}
    ).json()
    assert plain["results"][0]["case_id"] == "a.jpg"
    assert reranked["reranked"] is True
    assert reranked["results"][0]["case_id"] == "c.jpg"
    assert len(reranked["results"]) == 2


def test_text_search_out_of_domain_skips_retrieval(
    client: TestClient, gate: FakeGate, encoder: FakeEncoder
) -> None:
    gate.value = 0.5
    body = client.post("/search/text", json={"query": "chocolate cake recipe"}).json()
    assert body["status"] == "domain_out"
    assert body["results"] == []
    assert body["gate"]["status"] == "out_of_domain"
    assert body["message"]
    assert encoder.calls == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "x" * 501},
        {"query": "lung", "top_k": 0},
        {"query": "lung", "top_k": 51},
        {"query": "lung", "threshold": 0.1},
    ],
)
def test_text_search_rejects_invalid_requests(client: TestClient, payload: dict) -> None:
    assert client.post("/search/text", json=payload).status_code == 422


# --- /search/image -----------------------------------------------------------


def test_image_search_returns_cases_without_gate(client: TestClient) -> None:
    r = client.post(
        "/search/image", files={"file": ("q.png", _png(), "image/png")}, data={"top_k": "2"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["kind"] == "image"
    assert body["status"] == "ok"
    assert [c["case_id"] for c in body["results"]] == ["a.jpg", "b.jpg"]
    assert body["gate"]["status"] == "not_applicable"
    assert body["gate"]["score"] is None
    assert body["note"]


def test_image_search_rejects_wrong_content_type(client: TestClient) -> None:
    r = client.post("/search/image", files={"file": ("q.gif", b"GIF89a", "image/gif")})
    assert r.status_code == 415


def test_image_search_rejects_corrupt_image(client: TestClient) -> None:
    r = client.post("/search/image", files={"file": ("q.png", b"not an image", "image/png")})
    assert r.status_code == 400


def test_image_search_rejects_oversized_upload(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_app, "MAX_UPLOAD_BYTES", 10)
    r = client.post("/search/image", files={"file": ("q.png", _png(), "image/png")})
    assert r.status_code == 413


# --- /cases/{id} ---------------------------------------------------------------


def test_case_lookup_returns_caption(client: TestClient) -> None:
    body = client.get("/cases/c.jpg").json()
    assert body == {
        "case_id": "c.jpg",
        "caption": "lung adenocarcinoma biopsy",
        "pathology": "pulmonary",
        "disclaimer": DISCLAIMER,
    }


def test_case_lookup_unknown_is_404(client: TestClient) -> None:
    assert client.get("/cases/zzz.jpg").status_code == 404


@pytest.mark.parametrize("bad", ["..", ".hidden", "a b.jpg"])
def test_case_lookup_rejects_non_basename_ids(client: TestClient, bad: str) -> None:
    assert client.get(f"/cases/{bad}").status_code in (404, 422)
    assert client.get(f"/cases/{bad}").status_code != 200


# --- load_service ------------------------------------------------------------


def test_load_service_fails_loudly_without_artifacts(tmp_path: Path) -> None:
    settings = load_settings(
        {
            "PATHOSEARCH_ARTIFACTS_DIR": str(tmp_path / "artifacts"),
            "PATHOSEARCH_DATA_DIR": str(tmp_path / "data"),
        }
    )
    with pytest.raises(DataNotFoundError, match="PATHOSEARCH_ARTIFACTS_DIR"):
        load_service(settings, THRESHOLD)
