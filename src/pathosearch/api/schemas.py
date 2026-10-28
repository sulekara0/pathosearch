"""Pydantic request/response models for the search API."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

DISCLAIMER = "Research/education use only. Not a diagnostic tool."
GATE_CALIBRATION_NOTE = (
    "Domain gate threshold is not calibrated: chosen by hand on 10 queries, no ROC/PR "
    "analysis. Borderline queries may be misclassified. See docs/adr/0005-product-model-quiltnet.md."
)
IMAGE_SEARCH_NOTE = "Image-to-image retrieval quality has not been measured."

MAX_QUERY_CHARS = 500
MAX_TOP_K = 50


class TextSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    top_k: int = Field(default=10, ge=1, le=MAX_TOP_K)
    rerank: bool = False

    @field_validator("query")
    @classmethod
    def _strip_and_require_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must contain non-whitespace text")
        return v


class CaseResult(BaseModel):
    case_id: str
    score: float
    caption: str
    pathology: str


class Timing(BaseModel):
    encode_ms: float
    search_ms: float
    total_ms: float


class GateDecision(BaseModel):
    status: Literal["in_domain", "out_of_domain", "not_applicable"]
    score: float | None
    threshold: float | None
    calibrated: bool = False


class SearchResponse(BaseModel):
    query_id: UUID
    kind: Literal["text", "image"]
    status: Literal["ok", "domain_out", "empty"]
    message: str | None
    gate: GateDecision
    results: list[CaseResult]
    reranked: bool
    timing: Timing
    model: str
    disclaimer: str = DISCLAIMER
    note: str | None = None


class CaseResponse(BaseModel):
    case_id: str
    caption: str
    pathology: str
    disclaimer: str = DISCLAIMER


class GateInfo(BaseModel):
    threshold: float
    calibrated: bool = False
    note: str = GATE_CALIBRATION_NOTE


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model: str
    n_vectors: int
    gate: GateInfo
    startup_ms: dict[str, float]
