"""File management API routes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from data_processor.core.data_loader import DataLoader

from ..dependencies import get_app_state, get_data_loader, get_loaded_file
from ..schemas.file_schemas import (
    FileInfo,
    FileListResponse,
    FileUploadResponse,
    SignalInfo,
    SignalListResponse,
)
from ..state import AppState, LoadedFile

router = APIRouter()
logger = logging.getLogger(__name__)


class LoadFileRequest(BaseModel):
    """Request to load a file."""

    path: str = Field(description="Path to the file to load")


@router.post("/load", response_model=FileUploadResponse)
def load_file(
    request: LoadFileRequest,
    state: AppState = Depends(get_app_state),
    loader: DataLoader = Depends(get_data_loader),
) -> FileUploadResponse:
    """Load a CSV file into memory."""
    file_path = Path(request.path)

    # Validate file exists
    if not _file_exists(file_path):
        return _error_response("File not found")

    # Validate extension
    if not _is_valid_extension(file_path):
        return _error_response("Invalid file extension")

    # Load the file
    df = loader.load_csv_file(str(file_path), validate_security=True)
    if df is None:
        return _error_response("Failed to load file")

    # Get file size
    size_bytes = _get_file_size(file_path)

    # Add to state
    loaded_file = state.add_file(str(file_path), df, size_bytes)

    return FileUploadResponse(
        success=True,
        file_id=loaded_file.file_id,
        file_info=_create_file_info(loaded_file),
    )


@router.get("", response_model=FileListResponse)
def list_files(
    state: AppState = Depends(get_app_state),
) -> FileListResponse:
    """List all loaded files."""
    files = state.list_files()
    return FileListResponse(
        files=[_create_file_info(f) for f in files],
        total_count=len(files),
    )


@router.get("/{file_id}/signals", response_model=SignalListResponse)
def get_signals(
    loaded_file: LoadedFile = Depends(get_loaded_file),
) -> SignalListResponse:
    """Get list of signals from a loaded file."""
    df = loaded_file.dataframe
    signals = _extract_signal_info(df)
    numeric_count = sum(1 for s in signals if s.is_numeric)

    return SignalListResponse(
        file_id=loaded_file.file_id,
        signals=signals,
        numeric_count=numeric_count,
        total_count=len(signals),
    )


@router.delete("/{file_id}")
def delete_file(
    file_id: str,
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    """Delete a loaded file."""
    if not state.remove_file(file_id):
        raise HTTPException(status_code=404, detail=f"File not found: {file_id}")
    return {"success": True, "file_id": file_id}


# Helper functions - kept short and focused


def _file_exists(path: Path) -> bool:
    """Check if file exists."""
    return path.exists() and path.is_file()


def _is_valid_extension(path: Path) -> bool:
    """Check if file has valid extension."""
    valid_extensions = {".csv", ".txt", ".parquet"}
    return path.suffix.lower() in valid_extensions


def _get_file_size(path: Path) -> int:
    """Get file size in bytes."""
    return path.stat().st_size


def _error_response(message: str) -> FileUploadResponse:
    """Create an error response."""
    return FileUploadResponse(
        success=False,
        file_id="",
        error=message,
    )


def _create_file_info(loaded_file: LoadedFile) -> FileInfo:
    """Create FileInfo from LoadedFile."""
    return FileInfo(
        filename=loaded_file.filename,
        path=loaded_file.path,
        size_bytes=loaded_file.size_bytes,
        row_count=loaded_file.row_count,
        column_count=loaded_file.column_count,
        loaded_at=loaded_file.loaded_at,
    )


def _extract_signal_info(df: Any) -> list[SignalInfo]:
    """Extract signal information from dataframe."""
    signals = []
    for col in df.columns:
        col_data = df[col]
        is_numeric = np.issubdtype(col_data.dtype, np.number)
        signal = SignalInfo(
            name=str(col),
            dtype=str(col_data.dtype),
            is_numeric=is_numeric,
            non_null_count=int(col_data.notna().sum()),
            min_value=float(col_data.min()) if is_numeric else None,
            max_value=float(col_data.max()) if is_numeric else None,
        )
        signals.append(signal)
    return signals
