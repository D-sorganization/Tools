"""Python wrapper for the Rust/Polars bulk-I/O engine (issue #2989, Phase 2).

Tries to import the native ``data_processor_core`` PyO3 extension built by
maturin.  If the wheel is not present (e.g. during CI on plain CPython without
a compiled wheel), every function falls back to a pure-pandas implementation
so contract tests can run without a Rust toolchain.

Public functions match the contract defined in the epic:

- ``inspect(path) -> SchemaInfo``
- ``preview(path, nrows=100, columns=None) -> pd.DataFrame``
- ``convert(src, dst, format) -> ConversionReport``
- ``scan_batch(path, batch_size, columns=None) -> Iterator[pd.DataFrame]``
- ``filter_export(path, dst, predicate, columns=None) -> int``
- ``cancel()`` — signal cancellation (no-op for pandas fallback)

All paths must be ``str`` or ``pathlib.Path``-like.  ``ValueError`` is raised
for contract violations; ``NotImplementedError`` for operations not yet
available in the selected backend.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Generator, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pandas as pd

logger = logging.getLogger(__name__)


class DataProcessorRustError(RuntimeError):
    """Raised when the Rust data engine rejects or cannot complete a request."""


# ── Try to import the native extension ───────────────────────────────────────

_RUST_AVAILABLE = False
_rust_mod = None

try:
    import data_processor_core as _rust_mod_lib

    _rust_mod = _rust_mod_lib

    _RUST_AVAILABLE = True
    logger.debug("data_processor_core native extension loaded")
except ImportError:
    logger.debug(
        "data_processor_core native extension not found; using pandas fallback"
    )

# ── Public data types ─────────────────────────────────────────────────────────


@dataclass
class SchemaInfo:
    """Column schema and file metadata returned by :func:`inspect`."""

    columns: list[str] = field(default_factory=list)
    column_types: dict[str, str] = field(default_factory=dict)
    row_count_estimate: int = 0
    file_size_bytes: int = 0
    format: str = ""


@dataclass
class ConversionReport:
    """Statistics returned by :func:`convert`."""

    source: str = ""
    destination: str = ""
    output_format: str = ""
    rows_written: int = 0
    columns: list[str] = field(default_factory=list)
    bytes_written: int = 0


@dataclass(frozen=True)
class DatasetMetadata:
    """Compatibility metadata returned by :class:`RustBulkDataEngine`."""

    format: str
    row_count: int
    columns: list[str]
    byte_size: int


@dataclass(frozen=True)
class PreviewTable:
    """Compatibility preview returned by :class:`RustBulkDataEngine`."""

    columns: list[str]
    rows: list[dict[str, str]]
    rows_returned: int


@dataclass(frozen=True)
class BulkConversionReport:
    """Compatibility conversion report returned by :class:`RustBulkDataEngine`."""

    input: str
    output: str
    output_format: str
    rows_read: int
    rows_written: int
    columns: list[str]
    bytes_written: int


class RustBulkDataEngine:
    """Compatibility facade over the pandas fallback bulk-data functions."""

    def __init__(self, *, repo_root: Path | None = None) -> None:
        self.repo_root = repo_root

    @classmethod
    def from_repo_root(cls, repo_root: Path | None = None) -> RustBulkDataEngine:
        """Construct the compatibility engine using the current repository layout."""
        return cls(repo_root=repo_root)

    def is_available(self) -> bool:
        """Return whether this process can service bulk-data requests."""
        return True

    def inspect(self, path: str | os.PathLike[str]) -> DatasetMetadata:
        """Inspect a supported dataset through the fallback implementation."""
        info = inspect(path)
        return DatasetMetadata(
            format=info.format,
            row_count=info.row_count_estimate,
            columns=info.columns,
            byte_size=info.file_size_bytes,
        )

    def preview(
        self,
        path: str | os.PathLike[str],
        *,
        rows: int = 100,
        columns: list[str] | None = None,
    ) -> PreviewTable:
        """Preview rows through the fallback implementation."""
        frame = preview(path, nrows=rows, columns=columns)
        return PreviewTable(
            columns=[str(column) for column in frame.columns],
            rows=[
                {str(key): str(value) for key, value in row.items()}
                for row in frame.to_dict(orient="records")
            ],
            rows_returned=len(frame),
        )

    def convert(
        self,
        input_path: str | os.PathLike[str],
        output_path: str | os.PathLike[str],
        *,
        output_format: str = "csv",
        columns: list[str] | None = None,
    ) -> BulkConversionReport:
        """Convert a dataset through the fallback implementation."""
        if columns is not None:
            frame = preview(input_path, nrows=10**12, columns=columns)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            if output_format == "csv":
                frame.to_csv(output_path, index=False)
            elif output_format == "parquet":
                frame.to_parquet(output_path, index=False)
            else:
                raise ValueError(
                    f"unsupported output format '{output_format}': "
                    "only csv and parquet are supported"
                )
            rows_written = len(frame)
            bytes_written = Path(output_path).stat().st_size
            output_columns = [str(column) for column in frame.columns]
        else:
            report = convert(input_path, output_path, output_format)
            rows_written = report.rows_written
            bytes_written = report.bytes_written
            output_columns = report.columns

        return BulkConversionReport(
            input=str(input_path),
            output=str(output_path),
            output_format=output_format,
            rows_read=rows_written,
            rows_written=rows_written,
            columns=output_columns,
            bytes_written=bytes_written,
        )


# ── Cancellation token ────────────────────────────────────────────────────────

_cancelled: bool = False


def cancel() -> None:
    """Signal that the current operation should be cancelled.

    The pandas fallback checks this flag at batch boundaries.  The native Rust
    engine exposes a cancellation token in Phase 3.
    """
    global _cancelled  # noqa: PLW0603
    _cancelled = True
    logger.info("Engine cancellation requested")


def _reset_cancel() -> None:
    """Internal: reset the cancellation flag (called at the start of each op)."""
    global _cancelled  # noqa: PLW0603
    _cancelled = False


# ── Helpers ───────────────────────────────────────────────────────────────────


def _str_path(path: str | os.PathLike[str]) -> str:
    s = os.fspath(path) if isinstance(path, os.PathLike) else path
    return str(Path(s)) if s else ""


def _require_path(path: str) -> None:
    if not path:
        raise ValueError("path must not be empty")
    if not Path(path).is_file():
        raise FileNotFoundError(f"file does not exist: {path}")


def _detect_format(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".parquet":
        return "parquet"
    raise ValueError(
        f"unsupported format '{suffix}': only csv and parquet are supported"
    )


def _validate_output_format(fmt: str) -> None:
    if fmt not in ("csv", "parquet"):
        raise ValueError(
            f"unsupported output format '{fmt}': only csv and parquet are supported"
        )


def _select_columns(df: pd.DataFrame, columns: Sequence[str] | None) -> pd.DataFrame:
    if columns is None:
        return df
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"column not found: {missing[0]}")
    return df[list(columns)]


# ── inspect ───────────────────────────────────────────────────────────────────


def inspect(path: str | os.PathLike[str]) -> SchemaInfo:
    """Return column names, inferred types, row count estimate, and file size.

    Preconditions:
        - ``path`` must be a non-empty string or Path-like.
        - The file must exist and be readable.
        - Format must be ``csv`` or ``parquet``.

    Raises:
        ValueError: if path is empty or format is unsupported.
        FileNotFoundError: if the file does not exist.
    """
    _reset_cancel()
    p = _str_path(path)
    _require_path(p)
    fmt = _detect_format(p)

    if _RUST_AVAILABLE:
        raw: dict = cast(Any, _rust_mod).py_inspect(p)
        return SchemaInfo(
            columns=raw["columns"],
            column_types=raw["column_types"],
            row_count_estimate=raw["row_count_estimate"],
            file_size_bytes=raw["file_size_bytes"],
            format=raw["format"],
        )

    # ── pandas fallback ──────────────────────────────────────────────────────
    file_size_bytes = Path(p).stat().st_size

    if fmt == "csv":
        df_header = pd.read_csv(p, nrows=0)
        columns_list = list(df_header.columns)
        # Full scan for row count (cheap for CI test fixtures)
        df_full = pd.read_csv(p)
        row_count = len(df_full)
        column_types = {col: str(dtype) for col, dtype in df_full.dtypes.items()}
    else:  # parquet
        df_full = pd.read_parquet(p)
        columns_list = list(df_full.columns)
        row_count = len(df_full)
        column_types = {col: str(dtype) for col, dtype in df_full.dtypes.items()}

    return SchemaInfo(
        columns=columns_list,
        column_types=column_types,
        row_count_estimate=row_count,
        file_size_bytes=file_size_bytes,
        format=fmt,
    )


# ── preview ───────────────────────────────────────────────────────────────────


def preview(
    path: str | os.PathLike[str],
    nrows: int = 100,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return the first ``nrows`` rows as a DataFrame.

    Preconditions:
        - ``path`` must exist and be a supported format.
        - ``nrows`` must be greater than zero.
        - ``columns``, when provided, must all be present in the file.

    Raises:
        ValueError: for contract violations.
        FileNotFoundError: if the file does not exist.
    """
    _reset_cancel()
    p = _str_path(path)
    _require_path(p)
    fmt = _detect_format(p)
    if nrows <= 0:
        raise ValueError("nrows must be greater than zero")

    if _RUST_AVAILABLE:
        rows: list[dict] = cast(Any, _rust_mod).py_preview(
            p, nrows, list(columns) if columns is not None else None
        )
        return pd.DataFrame(rows, columns=list(columns) if columns else None)

    # ── pandas fallback ──────────────────────────────────────────────────────
    if fmt == "csv":
        df = pd.read_csv(p, nrows=nrows)
    else:
        df = pd.read_parquet(p).head(nrows)

    return _select_columns(df, columns)


# ── convert ───────────────────────────────────────────────────────────────────


def convert(
    src: str | os.PathLike[str],
    dst: str | os.PathLike[str],
    format: str,  # noqa: A002
) -> ConversionReport:
    """Convert ``src`` to ``dst`` in the given ``format``.

    Preconditions:
        - ``src`` must exist and be a supported format.
        - ``format`` must be ``"csv"`` or ``"parquet"``.

    Raises:
        ValueError: for contract violations or unsupported format combinations.
        FileNotFoundError: if ``src`` does not exist.
        NotImplementedError: for format paths not yet implemented.
    """
    _reset_cancel()
    p_src = _str_path(src)
    p_dst = _str_path(dst)
    _require_path(p_src)
    _detect_format(p_src)
    _validate_output_format(format)

    if _RUST_AVAILABLE:
        raw: dict = cast(Any, _rust_mod).py_convert(p_src, p_dst, format)
        return ConversionReport(
            source=raw["source"],
            destination=raw["destination"],
            output_format=raw["output_format"],
            rows_written=raw["rows_written"],
            columns=raw["columns"],
            bytes_written=raw["bytes_written"],
        )

    # ── pandas fallback ──────────────────────────────────────────────────────
    src_fmt = _detect_format(p_src)

    if src_fmt == "csv":
        df = pd.read_csv(p_src)
    else:
        df = pd.read_parquet(p_src)

    Path(p_dst).parent.mkdir(parents=True, exist_ok=True)

    if format == "csv":
        df.to_csv(p_dst, index=False)
    elif format == "parquet":
        df.to_parquet(p_dst, index=False)

    bytes_written = Path(p_dst).stat().st_size
    return ConversionReport(
        source=p_src,
        destination=p_dst,
        output_format=format,
        rows_written=len(df),
        columns=list(df.columns),
        bytes_written=bytes_written,
    )


# ── scan_batch ────────────────────────────────────────────────────────────────


def scan_batch(
    path: str | os.PathLike[str],
    batch_size: int,
    columns: Sequence[str] | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield DataFrames of ``batch_size`` rows until the file is exhausted.

    Preconditions:
        - ``path`` must exist and be a supported format.
        - ``batch_size`` must be greater than zero.

    Raises:
        ValueError: for contract violations.
        FileNotFoundError: if ``path`` does not exist.

    Yields:
        pd.DataFrame: successive batches of at most ``batch_size`` rows.
    """
    _reset_cancel()
    p = _str_path(path)
    _require_path(p)
    fmt = _detect_format(p)
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    if _RUST_AVAILABLE:
        # Phase 2: Rust scan_batch raises NotImplementedError — fall through to
        # the pandas fallback so CI can run contract tests without Phase 3.
        try:
            cast(Any, _rust_mod).py_scan_batch(
                p, batch_size, list(columns) if columns is not None else None
            )
            return
        except NotImplementedError:
            pass

    # ── pandas fallback ──────────────────────────────────────────────────────
    if fmt == "csv":
        reader: Generator[pd.DataFrame, None, None] = pd.read_csv(
            p, chunksize=batch_size
        )
        for chunk in reader:
            if _cancelled:
                logger.info("scan_batch cancelled after %d rows", len(chunk))
                return
            yield _select_columns(chunk, columns).reset_index(drop=True)
    else:
        df = pd.read_parquet(p)
        df = _select_columns(df, columns)
        for start in range(0, len(df), batch_size):
            if _cancelled:
                return
            yield df.iloc[start : start + batch_size].reset_index(drop=True)


# ── filter_export ─────────────────────────────────────────────────────────────


def filter_export(
    path: str | os.PathLike[str],
    dst: str | os.PathLike[str],
    predicate: str,
    columns: Sequence[str] | None = None,
) -> int:
    """Filter rows matching ``predicate`` (pandas query string) and export.

    Preconditions:
        - ``path`` must exist and be a supported format.
        - ``dst`` must be a non-empty path with a supported format extension.
        - ``predicate`` must be a non-empty string.

    Returns:
        Number of rows written to ``dst``.

    Raises:
        ValueError: for contract violations.
        FileNotFoundError: if ``path`` does not exist.
    """
    _reset_cancel()
    p = _str_path(path)
    p_dst = _str_path(dst)
    _require_path(p)
    if not p_dst:
        raise ValueError("dst must not be empty")
    if not predicate or not predicate.strip():
        raise ValueError("predicate must not be empty")
    fmt = _detect_format(p)
    dst_fmt = _detect_format(p_dst)

    if _RUST_AVAILABLE:
        try:
            return int(
                cast(Any, _rust_mod).py_filter_export(
                    p,
                    p_dst,
                    predicate,
                    list(columns) if columns is not None else None,
                )
            )
        except NotImplementedError:
            pass

    # ── pandas fallback ──────────────────────────────────────────────────────
    if fmt == "csv":
        df = pd.read_csv(p)
    else:
        df = pd.read_parquet(p)

    df = _select_columns(df, columns)
    filtered = df.query(predicate)

    Path(p_dst).parent.mkdir(parents=True, exist_ok=True)

    if dst_fmt == "csv":
        filtered.to_csv(p_dst, index=False)
    else:
        filtered.to_parquet(p_dst, index=False)

    return len(filtered)
