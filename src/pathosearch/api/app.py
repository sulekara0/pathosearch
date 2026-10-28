"""FastAPI app: text/image search, case lookup and health.

Run (real data, loads models at startup):
    uv run uvicorn pathosearch.api.app:create_app --factory

Tests pass a prebuilt ``SearchService`` to ``create_app`` so no model or data is needed.
Endpoints are sync ``def``: encoding is CPU-bound, FastAPI runs them in a threadpool.
"""

from __future__ import annotations

import io
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi import Path as PathParam
from PIL import Image, UnidentifiedImageError

from pathosearch.api.schemas import (
    IMAGE_SEARCH_NOTE,
    MAX_TOP_K,
    CaseResponse,
    CaseResult,
    GateDecision,
    GateInfo,
    HealthResponse,
    SearchResponse,
    TextSearchRequest,
    Timing,
)
from pathosearch.api.service import SearchService, load_service

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png"}
# Corpus ids are file basenames; reject anything that could be a path.
CASE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"


def create_app(service: SearchService | None = None) -> FastAPI:
    """Build the app. Without ``service``, load the real one from env at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if service is None:
            from pathosearch.config import load_api_settings, load_settings

            app.state.service = load_service(load_settings(), load_api_settings().gate_threshold)
        else:
            app.state.service = service
        yield

    app = FastAPI(
        title="PathoSearch API",
        description="Histopathology text/image search over Quilt-1M (QuiltNet + FAISS). "
        "Research/education use only; not a diagnostic tool.",
        version="0.1.0",
        lifespan=lifespan,
    )
    _register_routes(app)
    return app


def get_service(request: Request) -> SearchService:
    svc: SearchService = request.app.state.service
    return svc


Service = Annotated[SearchService, Depends(get_service)]


def _to_results(results: list[dict]) -> list[CaseResult]:
    return [
        CaseResult(
            case_id=Path(r["id"]).name,
            score=r["score"],
            caption=r["caption"],
            pathology=r["pathology"],
        )
        for r in results
    ]


def _ms_since(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)


def _read_image(upload: UploadFile) -> Image.Image:
    if upload.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(415, f"Unsupported content type {upload.content_type!r}; use JPEG/PNG.")
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Image larger than {MAX_UPLOAD_BYTES} bytes.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        return Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise HTTPException(400, "Uploaded file is not a valid image.") from exc


def _register_routes(app: FastAPI) -> None:
    @app.get("/health", response_model=HealthResponse)
    def health(svc: Service) -> HealthResponse:
        return HealthResponse(
            status="ok",
            model=svc.model_name,
            n_vectors=svc.n_vectors,
            gate=GateInfo(threshold=svc.gate_threshold),
            startup_ms=svc.startup_ms,
        )

    @app.post("/search/text", response_model=SearchResponse)
    def search_text(req: TextSearchRequest, svc: Service) -> SearchResponse:
        t0 = time.perf_counter()
        res = svc.search_text(req.query, req.top_k, req.rerank)
        in_domain = res["domain_score"] >= svc.gate_threshold
        return SearchResponse(
            query_id=uuid.uuid4(),
            kind="text",
            status=res["status"],  # type: ignore[arg-type]
            message=res["message"],
            gate=GateDecision(
                status="in_domain" if in_domain else "out_of_domain",
                score=res["domain_score"],
                threshold=svc.gate_threshold,
            ),
            results=_to_results(res["results"]),
            reranked=res["reranked"],
            timing=Timing(
                encode_ms=res["enc_ms"], search_ms=res["srch_ms"], total_ms=_ms_since(t0)
            ),
            model=svc.model_name,
        )

    @app.post("/search/image", response_model=SearchResponse)
    def search_image(
        svc: Service,
        file: Annotated[UploadFile, File(description="JPEG or PNG query image")],
        top_k: Annotated[int, Form(ge=1, le=MAX_TOP_K)] = 10,
    ) -> SearchResponse:
        t0 = time.perf_counter()
        image = _read_image(file)
        results, enc_ms, srch_ms = svc.search_image(image, top_k)
        return SearchResponse(
            query_id=uuid.uuid4(),
            kind="image",
            status="ok" if results else "empty",
            message=None,
            gate=GateDecision(status="not_applicable", score=None, threshold=None),
            results=_to_results(results),
            reranked=False,
            timing=Timing(encode_ms=enc_ms, search_ms=srch_ms, total_ms=_ms_since(t0)),
            model=svc.model_name,
            note=IMAGE_SEARCH_NOTE,
        )

    @app.get("/cases/{case_id}", response_model=CaseResponse)
    def get_case(
        case_id: Annotated[str, PathParam(pattern=CASE_ID_PATTERN, max_length=255)],
        svc: Service,
    ) -> CaseResponse:
        case = svc.get_case(case_id)
        if case is None:
            raise HTTPException(404, f"Unknown case id {case_id!r}.")
        return CaseResponse(**case)
