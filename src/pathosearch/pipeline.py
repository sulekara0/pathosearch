"""Text/image search pipeline shared by the UI and tests.

Text queries: domain gate -> (early exit | FAISS retrieval) -> optional
caption-based rerank -> status packaging. Image queries skip the text-anchor gate.

Dependencies (encoder, index, gate) are injected, so nothing here imports
Streamlit, torch or model weights; tests drive it with small fakes.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import faiss
import numpy as np

from pathosearch.faiss_search import raw_search
from pathosearch.reranker import rerank_results
from pathosearch.thresholds import GateResult, apply_domain_gate

# Rerank settings used by the app ("V5b"): pool = topk * RERANK_POOL_FACTOR.
RERANK_ALPHA = 0.75
RERANK_BETA = 0.25
RERANK_POOL_FACTOR = 5


class DomainScorer(Protocol):
    def score(self, query: str) -> float: ...


class TextEncoder(Protocol):
    def encode_texts(self, texts: list[str]) -> np.ndarray: ...


class ImageEncoder(Protocol):
    def encode_images(self, images: list[Any]) -> np.ndarray: ...


@dataclass(frozen=True)
class SearchIndex:
    """A loaded FAISS index plus the metadata needed to turn hits into results."""

    index: faiss.Index
    metric: str
    ids: list[str]
    captions: dict[str, dict]


class PipelineResult(GateResult):
    domain_score: float
    enc_ms: float
    srch_ms: float
    reranked: bool
    faiss_called: bool


def _timed_search(
    q_vec_fn: Callable[[], np.ndarray], idx: SearchIndex, topk: int
) -> tuple[list[dict], float, float]:
    t0 = time.perf_counter()
    q_vec = q_vec_fn()
    t1 = time.perf_counter()
    results = raw_search(idx.index, idx.metric, idx.ids, idx.captions, q_vec, topk)
    t2 = time.perf_counter()
    return results, (t1 - t0) * 1000, (t2 - t1) * 1000


def run_text_search(
    encoder: TextEncoder, idx: SearchIndex, query: str, topk: int
) -> tuple[list[dict], float, float]:
    """Encode ``query`` and search. Returns (results, encode_ms, search_ms)."""
    return _timed_search(lambda: encoder.encode_texts([query]), idx, topk)


def run_image_search(
    encoder: ImageEncoder, idx: SearchIndex, image: Any, topk: int
) -> tuple[list[dict], float, float]:
    """Encode a PIL image and search. Returns (results, encode_ms, search_ms)."""
    return _timed_search(lambda: encoder.encode_images([image]), idx, topk)


def search_pipeline(
    query: str,
    gate: DomainScorer,
    get_retrieval: Callable[[], tuple[TextEncoder, SearchIndex]],
    topk: int,
    enable_rerank: bool,
    domain_threshold: float,
) -> PipelineResult:
    """Run the text search pipeline.

    ``get_retrieval`` is called only for in-domain queries, so an out-of-domain
    query never loads the retrieval encoder/index or touches FAISS.
    """
    domain_score = gate.score(query)

    if domain_score < domain_threshold:
        gated = apply_domain_gate([], domain_score, domain_threshold)
        return PipelineResult(
            **gated,
            domain_score=domain_score,
            enc_ms=0.0,
            srch_ms=0.0,
            reranked=False,
            faiss_called=False,
        )

    encoder, idx = get_retrieval()
    fetch_k = topk * RERANK_POOL_FACTOR if enable_rerank else topk
    results, enc_ms, srch_ms = run_text_search(encoder, idx, query, fetch_k)

    if enable_rerank:
        results = rerank_results(query, results, topk=topk, alpha=RERANK_ALPHA, beta=RERANK_BETA)
    else:
        results = results[:topk]

    gated = apply_domain_gate(results, domain_score, domain_threshold)
    return PipelineResult(
        **gated,
        domain_score=domain_score,
        enc_ms=enc_ms,
        srch_ms=srch_ms,
        reranked=enable_rerank,
        faiss_called=True,
    )
