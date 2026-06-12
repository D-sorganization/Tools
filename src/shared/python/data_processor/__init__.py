"""Bulk-I/O engine package for the Data Processor (issue #2989).

Phase 2: engine contract definition and Python wrapper with pandas fallback.

Public API::

    from shared.python.data_processor import rust_engine

    info   = rust_engine.inspect("data.csv")
    df     = rust_engine.preview("data.csv", nrows=100)
    report = rust_engine.convert("in.csv", "out.parquet", format="parquet")
"""

from pathlib import Path

from .rust_engine import (
    ConversionReport,
    DataProcessorRustError,
    RustBulkDataEngine,
    SchemaInfo,
    convert,
    filter_export,
    inspect,
    preview,
    scan_batch,
)

__all__ = [
    "ConversionReport",
    "DataProcessorRustError",
    "RustBulkDataEngine",
    "SchemaInfo",
    "convert",
    "filter_export",
    "inspect",
    "preview",
    "scan_batch",
]

_full_package = (
    Path(__file__).resolve().parents[3]
    / "data_processing"
    / "data_processor"
    / "python"
    / "data_processor"
)
if _full_package.is_dir():
    __path__.append(str(_full_package))
