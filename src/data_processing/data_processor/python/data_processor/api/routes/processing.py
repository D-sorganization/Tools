"""Signal processing API routes."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from data_processor.core.signal_processor import SignalProcessor
from data_processor.models.processing_config import FilterConfig
from fastapi import APIRouter, Depends

from ..dependencies import get_app_state, get_loaded_file, get_signal_processor
from ..schemas.processing_schemas import (
    DataPreviewRequest,
    DataPreviewResponse,
    FilterRequest,
    FilterResponse,
    ProcessingStatus,
    SignalStatistics,
    StatisticsRequest,
    StatisticsResponse,
)
from ..state import AppState, LoadedFile

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/filter", response_model=FilterResponse)
def apply_filter(
    request: FilterRequest,
    state: AppState = Depends(get_app_state),
    processor: SignalProcessor = Depends(get_signal_processor),
) -> FilterResponse:
    """Apply a filter to signals in a loaded file."""
    # Get loaded file
    loaded_file = state.get_file(request.file_id)
    if loaded_file is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="File not found")

    df = loaded_file.dataframe
    signals = _determine_signals_to_process(df, request.signals)

    # Build filter config
    filter_config = _build_filter_config(request)

    # Apply filter
    try:
        filtered_df = _apply_filter_to_dataframe(processor, df, signals, filter_config)
        loaded_file.dataframe = filtered_df
        return _success_filter_response(request, signals, len(filtered_df))
    except Exception as e:
        logger.exception("Filter application failed")
        return _error_filter_response(request, str(e))


@router.post("/statistics", response_model=StatisticsResponse)
def calculate_statistics(
    request: StatisticsRequest,
    loaded_file: LoadedFile = Depends(get_loaded_file),
) -> StatisticsResponse:
    """Calculate statistics for signals."""
    df = loaded_file.dataframe
    signals = _determine_signals_to_process(df, request.signals)
    stats = [_calculate_signal_stats(df, signal) for signal in signals]

    return StatisticsResponse(
        file_id=request.file_id,
        statistics=stats,
    )


@router.post("/preview", response_model=DataPreviewResponse)
def preview_data(
    request: DataPreviewRequest,
    loaded_file: LoadedFile = Depends(get_loaded_file),
) -> DataPreviewResponse:
    """Preview data from a loaded file."""
    df = loaded_file.dataframe

    # Select columns
    columns = _select_columns(df, request.signals)
    subset = df[columns] if columns else df

    # Apply pagination
    paginated = _paginate_dataframe(subset, request.offset, request.limit)

    return DataPreviewResponse(
        file_id=request.file_id,
        columns=list(paginated.columns),
        data=_dataframe_to_list(paginated),
        total_rows=len(df),
        offset=request.offset,
        limit=len(paginated),
    )


# Helper functions - kept short and focused


def _determine_signals_to_process(df: Any, requested: list[str]) -> list[str]:
    """Determine which signals to process."""
    if requested:
        return [s for s in requested if s in df.columns]
    return df.select_dtypes(include=np.number).columns.tolist()


def _build_filter_config(request: FilterRequest) -> FilterConfig:
    """Build FilterConfig from request."""
    params = request.parameters.to_dict()
    return FilterConfig(filter_type=request.filter_type.value, parameters=params)


def _apply_filter_to_dataframe(
    processor: SignalProcessor,
    df: Any,
    signals: list[str],
    config: FilterConfig,
) -> Any:
    """Apply filter to selected signals in dataframe."""
    subset = df[signals].copy()
    filtered = processor.apply_filter(subset, config)
    result = df.copy()
    result[signals] = filtered[signals]
    return result


def _success_filter_response(
    request: FilterRequest, signals: list[str], row_count: int
) -> FilterResponse:
    """Create successful filter response."""
    return FilterResponse(
        success=True,
        status=ProcessingStatus.COMPLETED,
        file_id=request.file_id,
        filter_type=request.filter_type.value,
        signals_processed=signals,
        row_count=row_count,
    )


def _error_filter_response(request: FilterRequest, error: str) -> FilterResponse:
    """Create error filter response."""
    return FilterResponse(
        success=False,
        status=ProcessingStatus.FAILED,
        file_id=request.file_id,
        filter_type=request.filter_type.value,
        error=error,
    )


def _calculate_signal_stats(df: Any, signal: str) -> SignalStatistics:
    """Calculate statistics for a single signal."""
    col = df[signal]
    return SignalStatistics(
        name=signal,
        count=int(col.notna().sum()),
        mean=_safe_float(col.mean()),
        std=_safe_float(col.std()),
        min=_safe_float(col.min()),
        max=_safe_float(col.max()),
        median=_safe_float(col.median()),
        q25=_safe_float(col.quantile(0.25)),
        q75=_safe_float(col.quantile(0.75)),
    )


def _safe_float(value: Any) -> float | None:
    """Safely convert to float, handling NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return float(value)


def _select_columns(df: Any, signals: list[str]) -> list[str]:
    """Select columns from dataframe."""
    if signals:
        return [s for s in signals if s in df.columns]
    return list(df.columns)


def _paginate_dataframe(df: Any, offset: int, limit: int) -> Any:
    """Apply pagination to dataframe."""
    return df.iloc[offset : offset + limit]


def _dataframe_to_list(df: Any) -> list[list[Any]]:
    """Convert dataframe to list of lists."""
    return df.values.tolist()
