"""Pure-Python data preview service for the Sidekick Data Explorer tab."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .registry import WorkspaceRegistry, WorkspaceVariable

DEFAULT_DATA_EXPLORER_PREVIEW_ROWS = 20
DEFAULT_DATA_EXPLORER_MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024
SUPPORTED_DATA_EXPLORER_SUFFIXES = {
    ".csv": "csv",
    ".json": "json",
    ".parquet": "parquet",
    ".tsv": "tsv",
    ".xls": "excel",
    ".xlsx": "excel",
}


@dataclass(frozen=True, slots=True)
class DataExplorerColumnSummary:
    """Schema and null-count summary for one previewed column."""

    name: str
    dtype: str
    missing_count: int


@dataclass(frozen=True, slots=True)
class DataExplorerPreview:
    """Bounded preview payload returned by :class:`DataExplorerService`."""

    source_path: str
    format: str
    total_rows: int | None
    total_columns: int
    columns: tuple[DataExplorerColumnSummary, ...]
    preview_rows: tuple[dict[str, Any], ...]
    truncated: bool
    load_mode: str


class DataExplorerError(ValueError):
    """Structured user-facing error for preview/export failures."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class DataExplorerService:
    """Validate and preview project-scoped data files for Sidekick."""

    def __init__(
        self,
        *,
        project_root: str | Path,
        preview_rows: int = DEFAULT_DATA_EXPLORER_PREVIEW_ROWS,
        max_file_size_bytes: int = DEFAULT_DATA_EXPLORER_MAX_FILE_SIZE_BYTES,
    ) -> None:
        if preview_rows < 1:
            raise ValueError("preview_rows must be positive")
        if max_file_size_bytes < 1:
            raise ValueError("max_file_size_bytes must be positive")
        self._project_root = Path(project_root).expanduser().resolve()
        self.preview_rows = preview_rows
        self.max_file_size_bytes = max_file_size_bytes

    def preview_file(self, path: str | Path) -> DataExplorerPreview:
        """Return a bounded preview for one project-scoped data file."""
        resolved = self._resolve_project_file(path)
        format_name = _format_name_for(resolved)
        file_size = resolved.stat().st_size

        if file_size > self.max_file_size_bytes and format_name not in {"csv", "tsv"}:
            raise DataExplorerError(
                "file_too_large",
                (
                    f"{resolved.name} exceeds the "
                    f"{self.max_file_size_bytes} byte preview limit."
                ),
            )

        load_mode = "full"
        try:
            if file_size > self.max_file_size_bytes and format_name in {"csv", "tsv"}:
                frame = self._load_delimited_preview(resolved, format_name)
                total_rows = _count_delimited_rows(resolved)
                load_mode = "sampled"
            else:
                frame = self._load_full_frame(resolved, format_name)
                total_rows = len(frame.index)
        except DataExplorerError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize preview failures
            raise DataExplorerError(
                "read_failed",
                f"Could not preview {resolved.name}: {exc}",
            ) from exc

        preview_frame = frame.head(self.preview_rows)
        preview_rows = tuple(_frame_rows(preview_frame))
        return DataExplorerPreview(
            source_path=str(resolved),
            format=format_name,
            total_rows=total_rows,
            total_columns=len(frame.columns),
            columns=tuple(_column_summaries(frame)),
            preview_rows=preview_rows,
            truncated=(total_rows or 0) > len(preview_rows),
            load_mode=load_mode,
        )

    def export_selection(
        self,
        preview: DataExplorerPreview,
        registry: WorkspaceRegistry,
        variable_name: str,
        *,
        selected_columns: list[str] | None = None,
        row_limit: int | None = None,
    ) -> WorkspaceVariable:
        """Export a bounded row/column selection into the shared workspace."""
        name = variable_name.strip()
        if not name:
            raise DataExplorerError(
                "invalid_variable_name",
                "Workspace variable name must be non-empty.",
            )
        columns = _resolve_selected_columns(preview, selected_columns)
        rows = _bounded_rows(preview.preview_rows, row_limit)
        if len(columns) == 1:
            value: Any = [row.get(columns[0]) for row in rows]
        else:
            value = [{column: row.get(column) for column in columns} for row in rows]
        return registry.set(name, value)

    def build_data_processor_request(
        self,
        preview: DataExplorerPreview,
        *,
        selected_columns: list[str] | None = None,
        row_limit: int | None = None,
    ) -> dict[str, Any]:
        """Return a JSON-safe handoff payload for a future Data Processor host."""
        columns = _resolve_selected_columns(preview, selected_columns)
        effective_row_limit = row_limit if row_limit is not None else self.preview_rows
        return {
            "tool_id": "data_processor",
            "source_path": preview.source_path,
            "source_format": preview.format,
            "selected_columns": columns,
            "row_limit": effective_row_limit,
        }

    def _resolve_project_file(self, path: str | Path) -> Path:
        raw = Path(path).expanduser()
        resolved = (
            raw.resolve() if raw.is_absolute() else (self._project_root / raw).resolve()
        )
        if not resolved.exists() or not resolved.is_file():
            raise DataExplorerError(
                "file_not_found",
                f"Data file not found: {resolved}",
            )
        if self._project_root not in {resolved, *resolved.parents}:
            raise DataExplorerError(
                "path_outside_project",
                f"{resolved} is outside the scoped project root.",
            )
        return resolved

    def _load_delimited_preview(self, path: Path, format_name: str) -> pd.DataFrame:
        separator = "\t" if format_name == "tsv" else ","
        return pd.read_csv(path, sep=separator, nrows=self.preview_rows)

    def _load_full_frame(self, path: Path, format_name: str) -> pd.DataFrame:
        if format_name == "csv":
            return pd.read_csv(path)
        if format_name == "tsv":
            return pd.read_csv(path, sep="\t")
        if format_name == "json":
            return _json_frame(path)
        if format_name == "parquet":
            return pd.read_parquet(path)
        if format_name == "excel":
            return pd.read_excel(path)
        raise DataExplorerError(
            "unsupported_format",
            (
                f"{path.suffix.lower()} is not supported by the Data Explorer "
                "preview service."
            ),
        )


def _format_name_for(path: Path) -> str:
    try:
        return SUPPORTED_DATA_EXPLORER_SUFFIXES[path.suffix.lower()]
    except KeyError as exc:
        raise DataExplorerError(
            "unsupported_format",
            (
                f"{path.suffix.lower()} is not supported by the Data Explorer "
                "preview service."
            ),
        ) from exc


def _json_frame(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, dict):
        scalar_mapping = {
            key: value
            for key, value in payload.items()
            if not isinstance(value, list | dict)
        }
        if scalar_mapping:
            return pd.DataFrame([scalar_mapping])
        return pd.DataFrame(payload)
    raise DataExplorerError(
        "unsupported_json_shape",
        "JSON previews require an object or list payload.",
    )


def _frame_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in frame.to_dict(orient="records"):
        rows.append(
            {str(key): _normalize_cell(value) for key, value in payload.items()}
        )
    return rows


def _normalize_dtype(dtype: Any) -> str:
    """Normalize pandas dtype names for stable cross-version summaries."""
    dtype_str = str(dtype).lower()
    # Normalize string-like dtype names across pandas versions
    if dtype_str in {"object", "string", "string[python]", "string[pyarrow]"}:
        return "object"
    # Normalize integer types
    if dtype_str in {
        "int64",
        "int32",
        "int16",
        "int8",
        "uint64",
        "uint32",
        "uint16",
        "uint8",
    }:
        return "int"
    # Normalize float types
    if dtype_str in {"float64", "float32", "float16"}:
        return "float"
    # Normalize boolean type
    if dtype_str in {"bool", "boolean"}:
        return "bool"
    # Normalize datetime types
    if "datetime" in dtype_str or "timedelta" in dtype_str:
        return "datetime"
    # Normalize category type
    if "category" in dtype_str:
        return "category"
    # Return original for other types
    return dtype_str


def _column_summaries(frame: pd.DataFrame) -> list[DataExplorerColumnSummary]:
    return [
        DataExplorerColumnSummary(
            name=str(column),
            dtype=_stable_dtype_name(frame[column].dtype),
            missing_count=int(frame[column].isna().sum()),
        )
        for column in frame.columns
    ]


def _stable_dtype_name(dtype: Any) -> str:
    dtype_name = str(dtype)
    if dtype_name in {"str", "string"}:
        return "object"
    return dtype_name


def _count_delimited_rows(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        row_count = sum(1 for _ in reader)
    return max(0, row_count - 1)


def _normalize_cell(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def _resolve_selected_columns(
    preview: DataExplorerPreview,
    selected_columns: list[str] | None,
) -> list[str]:
    available = [column.name for column in preview.columns]
    if not selected_columns:
        return available
    normalized = [column.strip() for column in selected_columns if column.strip()]
    missing = [column for column in normalized if column not in available]
    if missing:
        raise DataExplorerError(
            "unknown_columns",
            f"Selected columns are not in the preview schema: {missing}",
        )
    return normalized


def _bounded_rows(
    rows: tuple[dict[str, Any], ...],
    row_limit: int | None,
) -> list[dict[str, Any]]:
    if row_limit is None:
        return list(rows)
    if row_limit < 1:
        raise DataExplorerError("invalid_row_limit", "Row limit must be positive.")
    return list(rows[:row_limit])
