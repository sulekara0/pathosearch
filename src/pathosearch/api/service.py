"""Search service: the loaded gate, retrieval encoder and index behind the API.

``SearchService`` holds already-loaded dependencies, so tests build it from small
fakes. ``load_service`` builds the real one from settings (QuiltNet retrieval,
CLIP text-tower gate; ADR-0005) and records how long each startup stage took.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pathosearch.config import Settings, require_dir, require_file
from pathosearch.pipeline import (
    DomainScorer,
    PipelineResult,
    SearchIndex,
    run_image_search,
    search_pipeline,
)

logger = logging.getLogger(__name__)

PRODUCT_ENCODER = "quilt"
PRODUCT_MODEL_ID = "hf-hub:wisdomik/QuiltNet-B-32"
GATE_ENCODER = "clip"


class Encoder(Protocol):
    def encode_texts(self, texts: list[str]) -> Any: ...

    def encode_images(self, images: list[Any]) -> Any: ...


@dataclass
class SearchService:
    gate: DomainScorer
    encoder: Encoder
    index: SearchIndex
    gate_threshold: float
    model_name: str = PRODUCT_MODEL_ID
    startup_ms: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._case_ids = {Path(i).name for i in self.index.ids}

    @property
    def n_vectors(self) -> int:
        return len(self.index.ids)

    def search_text(self, query: str, top_k: int, rerank: bool) -> PipelineResult:
        return search_pipeline(
            query,
            self.gate,
            lambda: (self.encoder, self.index),
            top_k,
            rerank,
            self.gate_threshold,
        )

    def search_image(self, image: Any, top_k: int) -> tuple[list[dict], float, float]:
        return run_image_search(self.encoder, self.index, image, top_k)

    def get_case(self, case_id: str) -> dict[str, str] | None:
        """Caption/pathology for a corpus image id (file basename), or None if unknown."""
        if case_id not in self._case_ids:
            return None
        rec = self.index.captions.get(case_id, {})
        return {
            "case_id": case_id,
            "caption": rec.get("caption", ""),
            "pathology": rec.get("pathology", ""),
        }


@contextmanager
def _timed(timings: dict[str, float], stage: str) -> Iterator[None]:
    t0 = time.perf_counter()
    yield
    timings[stage] = round((time.perf_counter() - t0) * 1000, 1)
    logger.info("startup stage %s: %.1f ms", stage, timings[stage])


def load_service(settings: Settings, gate_threshold: float) -> SearchService:
    """Load index, captions, QuiltNet and the CLIP gate. Missing data raises, no fallback."""
    from pathosearch.faiss_search import load_captions, load_ids, load_index

    hint = (
        "Mount the QuiltNet artifacts and captions JSONL and set PATHOSEARCH_ARTIFACTS_DIR / "
        "PATHOSEARCH_CAPTIONS_PATH (see .env.example)."
    )
    art_dir = require_dir(settings.encoder_dir(PRODUCT_ENCODER), hint)
    captions_path = require_file(settings.captions_path, hint)

    timings: dict[str, float] = {}
    with _timed(timings, "index"):
        index, metric = load_index(str(art_dir))
    with _timed(timings, "ids"):
        ids = load_ids(str(art_dir))
    if index.ntotal != len(ids):
        raise ValueError(f"index has {index.ntotal} vectors but ids.json has {len(ids)} entries")
    with _timed(timings, "captions"):
        captions = load_captions(captions_path)

    from pathosearch.domain_gate import DomainGate
    from pathosearch.encoders import get_encoder

    with _timed(timings, "retrieval_encoder"):
        encoder = get_encoder(PRODUCT_ENCODER)
    with _timed(timings, "gate_encoder"):
        gate = DomainGate(get_encoder(GATE_ENCODER))
    timings["total"] = round(sum(timings.values()), 1)

    return SearchService(
        gate=gate,
        encoder=encoder,
        index=SearchIndex(index, metric, ids, captions),
        gate_threshold=gate_threshold,
        startup_ms=timings,
    )
