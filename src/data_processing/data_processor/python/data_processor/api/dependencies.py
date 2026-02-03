"""FastAPI dependencies for the API."""

from __future__ import annotations

from typing import TYPE_CHECKING

from data_processor.core.data_loader import DataLoader
from data_processor.core.signal_processor import SignalProcessor
from fastapi import Depends, HTTPException, Request

from .state import AppState, LoadedFile

if TYPE_CHECKING:
    pass


def get_app_state(request: Request) -> AppState:
    """Get the application state from the request."""
    return request.app.state.app_state


def get_data_loader() -> DataLoader:
    """Get a DataLoader instance."""
    return DataLoader(use_high_performance=False)


def get_signal_processor() -> SignalProcessor:
    """Get a SignalProcessor instance."""
    return SignalProcessor()


def get_loaded_file(
    file_id: str,
    state: AppState = Depends(get_app_state),
) -> LoadedFile:
    """Get a loaded file by ID, raising 404 if not found."""
    loaded_file = state.get_file(file_id)
    if loaded_file is None:
        raise HTTPException(status_code=404, detail=f"File not found: {file_id}")
    return loaded_file
