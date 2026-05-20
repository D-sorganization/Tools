"""Native bulk-data boundary helpers for :mod:`data_processor.core.data_loader`."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Protocol

import pandas as pd

from data_processor.contracts import require
from data_processor.rust_engine import DataProcessorRustError, RustBulkDataEngine

logger = logging.getLogger(__name__)


class NativeBulkHost(Protocol):
    """Minimal host contract required by the native bulk helpers."""

    _rust_engine: RustBulkDataEngine | None


def inspect_dataset(host: NativeBulkHost, file_path: str) -> dict[str, Any]:
    """Inspect a dataset through the native streaming engine."""
    require(
        isinstance(file_path, str) and bool(file_path.strip()),
        "file_path must be a non-empty string",
        file_path,
    )
    return asdict(_get_rust_engine(host).inspect(file_path))


def preview_dataset(
    host: NativeBulkHost,
    file_path: str,
    rows: int = 100,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Preview a dataset through the native streaming engine."""
    require(
        isinstance(file_path, str) and bool(file_path.strip()),
        "file_path must be a non-empty string",
        file_path,
    )
    require(rows > 0, "rows must be greater than zero", rows)
    preview = _get_rust_engine(host).preview(
        file_path,
        rows=rows,
        columns=columns,
    )
    return pd.DataFrame(preview.rows, columns=preview.columns)


def convert_dataset(
    host: NativeBulkHost,
    input_path: str,
    output_path: str,
    *,
    output_format: str = "csv",
    columns: list[str] | None = None,
) -> dict[str, Any]:
    """Convert a dataset through the native streaming engine."""
    require(
        isinstance(input_path, str) and bool(input_path.strip()),
        "input_path must be a non-empty string",
        input_path,
    )
    require(
        isinstance(output_path, str) and bool(output_path.strip()),
        "output_path must be a non-empty string",
        output_path,
    )
    report = _get_rust_engine(host).convert(
        input_path,
        output_path,
        output_format=output_format,
        columns=columns,
    )
    return asdict(report)


def _get_rust_engine(host: NativeBulkHost) -> RustBulkDataEngine:
    """Return the native engine, lazily constructing it for production use."""
    if host._rust_engine is None:
        try:
            host._rust_engine = RustBulkDataEngine.from_repo_root()
        except DataProcessorRustError:
            logger.exception("Rust bulk data engine is unavailable")
            raise
    return host._rust_engine
