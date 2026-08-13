"""Strict retained-record ingestion for the PyQt launch-monitor workspace."""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path

import pandas as pd

from rate_of_closure.launch_monitor_linked_scatter import MAX_RETAINED_ROWS
from rate_of_closure.launch_monitor_numeric import finite_launch_monitor_scalar

_SCALAR_TYPES = (str, int, float, bool, type(None))
_MAX_SAFE_INTEGER = 9_007_199_254_740_991
MAX_IMPORT_BYTES = 8 * 1024 * 1024
MAX_IMPORT_COLUMNS = 256
MAX_IMPORT_CELLS = 2_000_000


def _coerce_csv_cell(value: str) -> str | float | None:
    stripped = value.strip()
    if not stripped:
        return None
    numeric = finite_launch_monitor_scalar(stripped)
    if numeric is None or (numeric.is_integer() and abs(numeric) > _MAX_SAFE_INTEGER):
        return stripped
    return int(numeric) if numeric.is_integer() else numeric


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"JSON constant {value} is not permitted")


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field name is not permitted: {key}")
        result[key] = value
    return result


def _json_scalar_is_portable(value: object) -> bool:
    if not isinstance(value, _SCALAR_TYPES):
        return False
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, int):
        return abs(value) <= _MAX_SAFE_INTEGER
    return math.isfinite(value) and (
        not value.is_integer() or abs(value) <= _MAX_SAFE_INTEGER
    )


def read_launch_monitor_frame(path: Path) -> pd.DataFrame:
    """Read a CSV/JSON record table with exact flat-row shape validation."""
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".json"}:
        raise ValueError("Launch-monitor import supports CSV and JSON")
    if path.stat().st_size > MAX_IMPORT_BYTES:
        raise ValueError(f"Launch-monitor import exceeds {MAX_IMPORT_BYTES} bytes")
    with path.open("rb") as stream:
        raw = stream.read(MAX_IMPORT_BYTES + 1)
    if len(raw) > MAX_IMPORT_BYTES:
        raise ValueError(f"Launch-monitor import exceeds {MAX_IMPORT_BYTES} bytes")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("Launch-monitor import must be valid UTF-8") from error
    if suffix == ".csv":
        rows: list[list[str]] = []
        try:
            for row in csv.reader(io.StringIO(text, newline=""), strict=True):
                if not any(value for value in row):
                    continue
                if len(row) > MAX_IMPORT_COLUMNS:
                    raise ValueError(
                        f"Launch-monitor import exceeds {MAX_IMPORT_COLUMNS} columns"
                    )
                rows.append(row)
                if len(rows) > MAX_RETAINED_ROWS + 1:
                    raise ValueError(
                        f"Launch-monitor import exceeds {MAX_RETAINED_ROWS} rows"
                    )
        except csv.Error as error:
            raise ValueError("Launch-monitor CSV is malformed") from error
        if len(rows) < 2:
            raise ValueError("CSV must contain a header and at least one row")
        headers = tuple(item.strip() for item in rows[0])
        if any(not item for item in headers) or len(set(headers)) != len(headers):
            raise ValueError("CSV headers must be non-empty and unique")
        if any(len(row) != len(headers) for row in rows[1:]):
            raise ValueError("Every CSV data row must match the header width")
        if len(rows[1:]) * len(headers) > MAX_IMPORT_CELLS:
            raise ValueError(
                f"Launch-monitor import exceeds {MAX_IMPORT_CELLS} dense cells"
            )
        return pd.DataFrame(
            [[_coerce_csv_cell(value) for value in row] for row in rows[1:]],
            columns=headers,
        )
    if suffix == ".json":
        try:
            payload = json.loads(
                text,
                parse_constant=_reject_nonstandard_json_constant,
                object_pairs_hook=_strict_object,
            )
        except RecursionError as error:
            raise ValueError("Launch-monitor JSON exceeds the nesting limit") from error
        if not isinstance(payload, list) or any(
            not isinstance(row, dict) for row in payload
        ):
            raise ValueError("JSON launch-monitor data must be an array of records")
        if len(payload) > MAX_RETAINED_ROWS:
            raise ValueError(f"Launch-monitor import exceeds {MAX_RETAINED_ROWS} rows")
        union = {key for row in payload for key in row}
        if len(union) > MAX_IMPORT_COLUMNS:
            raise ValueError(
                f"Launch-monitor import exceeds {MAX_IMPORT_COLUMNS} columns"
            )
        if len(payload) * len(union) > MAX_IMPORT_CELLS:
            raise ValueError(
                f"Launch-monitor import exceeds {MAX_IMPORT_CELLS} dense cells"
            )
        if any(not key.strip() for row in payload for key in row):
            raise ValueError("JSON launch-monitor field names must be non-empty")
        if any(
            not _json_scalar_is_portable(value)
            for row in payload
            for value in row.values()
        ):
            raise ValueError(
                "JSON launch-monitor record values must be portable finite scalars"
            )
        return pd.DataFrame.from_records(payload)
    raise AssertionError("validated launch-monitor suffix was not dispatched")


__all__ = ["read_launch_monitor_frame"]
