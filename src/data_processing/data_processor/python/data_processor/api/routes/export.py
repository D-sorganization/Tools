"""Data export API routes."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends

from data_processor.core.data_loader import DataLoader

from ..dependencies import get_app_state, get_data_loader, get_loaded_file
from ..schemas.processing_schemas import ExportRequest, ExportResponse
from ..state import AppState, LoadedFile

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("", response_model=ExportResponse)
def export_data(
    request: ExportRequest,
    loaded_file: LoadedFile = Depends(get_loaded_file),
    loader: DataLoader = Depends(get_data_loader),
) -> ExportResponse:
    """Export processed data to a file."""
    df = loaded_file.dataframe

    # Select signals if specified
    if request.signals:
        columns = [s for s in request.signals if s in df.columns]
        df = df[columns] if columns else df

    # Determine output path
    output_path = _determine_output_path(request, loaded_file)

    # Export data
    try:
        success = _export_dataframe(loader, df, output_path, request.format.value)
        if success:
            return _success_export_response(output_path, request.format.value)
        return _error_export_response(output_path, request.format.value, "Export failed")
    except Exception as e:
        logger.exception("Export failed")
        return _error_export_response(output_path, request.format.value, str(e))


# Also add this endpoint to the processing router for convenience
# This is registered in the processing router as well


def register_export_on_processing(router: APIRouter) -> None:
    """Register export endpoint on the processing router."""

    @router.post("/export", response_model=ExportResponse)
    def export_from_processing(
        request: ExportRequest,
        state: AppState = Depends(get_app_state),
        loader: DataLoader = Depends(get_data_loader),
    ) -> ExportResponse:
        """Export processed data to a file."""
        loaded_file = state.get_file(request.file_id)
        if loaded_file is None:
            return _error_export_response("", request.format.value, "File not found")

        df = loaded_file.dataframe
        if request.signals:
            columns = [s for s in request.signals if s in df.columns]
            df = df[columns] if columns else df

        output_path = _determine_output_path(request, loaded_file)
        try:
            success = _export_dataframe(loader, df, output_path, request.format.value)
            if success:
                return _success_export_response(output_path, request.format.value)
            return _error_export_response(
                output_path, request.format.value, "Export failed"
            )
        except Exception as e:
            logger.exception("Export failed")
            return _error_export_response(output_path, request.format.value, str(e))


# Helper functions - kept short and focused


def _determine_output_path(request: ExportRequest, loaded_file: LoadedFile) -> str:
    """Determine the output file path."""
    if request.filename:
        return request.filename
    return _generate_output_path(loaded_file.filename, request.format.value)


def _generate_output_path(original_filename: str, format_type: str) -> str:
    """Generate output path based on original filename."""
    stem = Path(original_filename).stem
    extension = _get_extension_for_format(format_type)
    temp_dir = tempfile.gettempdir()
    return str(Path(temp_dir) / f"{stem}_processed{extension}")


def _get_extension_for_format(format_type: str) -> str:
    """Get file extension for export format."""
    extensions = {
        "csv": ".csv",
        "excel": ".xlsx",
        "parquet": ".parquet",
        "hdf5": ".h5",
        "feather": ".feather",
    }
    return extensions.get(format_type, ".csv")


def _export_dataframe(
    loader: DataLoader, df: Any, output_path: str, format_type: str
) -> bool:
    """Export dataframe to file."""
    return loader.save_dataframe(df, output_path, format_type=format_type)


def _get_file_size(path: str) -> int:
    """Get file size in bytes."""
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0


def _success_export_response(output_path: str, format_type: str) -> ExportResponse:
    """Create successful export response."""
    return ExportResponse(
        success=True,
        filename=output_path,
        format=format_type,
        size_bytes=_get_file_size(output_path),
    )


def _error_export_response(
    output_path: str, format_type: str, error: str
) -> ExportResponse:
    """Create error export response."""
    return ExportResponse(
        success=False,
        filename=output_path,
        format=format_type,
        error=error,
    )
