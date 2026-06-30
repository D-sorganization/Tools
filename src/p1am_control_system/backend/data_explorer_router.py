"""FastAPI router for the Data Explorer analysis suite.

Exposes the dataset-build and analysis endpoints under ``/api/explorer``. Every
route is a thin adapter over :mod:`data_explorer_service`: it receives a
validated request model, calls the service, and returns a response model. The
numeric kernels enforce the rich Design-by-Contract preconditions and raise
``ValueError``/``TypeError``; this router maps both to ``HTTPException(400)`` so
malformed analysis requests surface as clean client errors rather than 500s.

The historian read here is read-only and already public via ``/api/trends`` and
``/api/export``, so — like those — these analysis endpoints carry no admin gate.

The router is created by :func:`create_data_explorer_router`, which takes the
backend's ``get_session`` dependency callable so the historian-backed endpoints
(``/signals``, ``/dataset``) get a request-scoped DB session. The integrator
wires this into ``main.py``; this module never imports ``main`` (LOD).
"""

from __future__ import annotations

from collections.abc import Callable

from data_explorer_enums import ExportFormat
from data_explorer_models import (
    ColumnsRequest,
    CorrelationRequest,
    CorrelationResponse,
    DatasetRequest,
    DatasetResponse,
    ExportRequest,
    HistogramRequest,
    HistogramResponse,
    PcaRequest,
    PcaResponse,
    SignalListResponse,
    SpectrumRequest,
    SpectrumResponse,
    StatisticsResponse,
    TrendlineRequest,
    TrendlineResponse,
)
from data_explorer_service import (
    build_dataset,
    compute_correlation,
    compute_histogram,
    compute_pca,
    compute_spectrum,
    compute_statistics,
    compute_trendline,
    dataset_to_csv_rows,
    dataset_to_json,
    list_signals,
    validate_export,
)
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

__all__ = ["create_data_explorer_router"]


def create_data_explorer_router(get_session_dep: Callable[..., object]) -> APIRouter:
    """Build the Data Explorer ``APIRouter`` bound to a session dependency.

    Args:
        get_session_dep: The backend's ``get_session`` FastAPI dependency
            (a generator yielding a SQLModel ``Session``). Used by the
            historian-backed ``/signals`` and ``/dataset`` routes.

    Returns:
        An :class:`fastapi.APIRouter` prefixed ``/api/explorer`` exposing every
        analysis endpoint with ``response_model`` set where applicable.
    """
    router = APIRouter(prefix="/api/explorer", tags=["explorer"])

    @router.get("/signals", response_model=SignalListResponse)
    async def get_signals(
        session: object = Depends(get_session_dep),  # noqa: B008
    ) -> SignalListResponse:
        try:
            return list_signals(session)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/dataset", response_model=DatasetResponse)
    async def post_dataset(
        req: DatasetRequest,
        session: object = Depends(get_session_dep),  # noqa: B008
    ) -> DatasetResponse:
        try:
            return build_dataset(session, req)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/statistics", response_model=StatisticsResponse)
    async def post_statistics(req: ColumnsRequest) -> StatisticsResponse:
        try:
            return compute_statistics(req)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/correlation", response_model=CorrelationResponse)
    async def post_correlation(req: CorrelationRequest) -> CorrelationResponse:
        try:
            return compute_correlation(req)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/spectrum", response_model=SpectrumResponse)
    async def post_spectrum(req: SpectrumRequest) -> SpectrumResponse:
        try:
            return compute_spectrum(req)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/trendline", response_model=TrendlineResponse)
    async def post_trendline(req: TrendlineRequest) -> TrendlineResponse:
        try:
            return compute_trendline(req)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/pca", response_model=PcaResponse)
    async def post_pca(req: PcaRequest) -> PcaResponse:
        try:
            return compute_pca(req)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/histogram", response_model=HistogramResponse)
    async def post_histogram(req: HistogramRequest) -> HistogramResponse:
        try:
            return compute_histogram(req)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/export", response_model=None)
    async def post_export(req: ExportRequest) -> StreamingResponse | JSONResponse:
        try:
            # Eagerly validate the index: the CSV path streams lazily, so a bad
            # value discovered mid-iteration could not become a 4xx.
            validate_export(req.index, req.columns)
            if req.format == ExportFormat.JSON:
                return JSONResponse(content=dataset_to_json(req.index, req.columns))
            filename = req.filename or "dataset.csv"
            return StreamingResponse(
                dataset_to_csv_rows(req.index, req.columns),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={filename}"},
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return router
