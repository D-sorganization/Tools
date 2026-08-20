"""Strict retained-record ingestion for the PyQt launch-monitor workspace."""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from typing import cast

import pandas as pd

from rate_of_closure.launch_monitor_numeric import finite_launch_monitor_scalar

_SCALAR_TYPES = (str, int, float, bool, type(None))
_MAX_SAFE_INTEGER = 9_007_199_254_740_991
MAX_IMPORT_BYTES = 8 * 1024 * 1024
MAX_IMPORT_FIELD_UTF8_BYTES = 64 * 1024
MAX_IMPORT_ROWS = 250_000
MAX_IMPORT_COLUMNS = 256
MAX_IMPORT_CELLS = 2_000_000


def _normalize_unicode_scalar(value: str) -> str:
    normalized: list[str] = []
    index = 0
    while index < len(value):
        codepoint = ord(value[index])
        if 0xD800 <= codepoint <= 0xDBFF:
            if index + 1 >= len(value):
                raise ValueError("Launch-monitor text must be well-formed Unicode")
            low = ord(value[index + 1])
            if not 0xDC00 <= low <= 0xDFFF:
                raise ValueError("Launch-monitor text must be well-formed Unicode")
            normalized.append(
                chr(0x10000 + ((codepoint - 0xD800) << 10) + low - 0xDC00)
            )
            index += 2
            continue
        if 0xDC00 <= codepoint <= 0xDFFF:
            raise ValueError("Launch-monitor text must be well-formed Unicode")
        normalized.append(value[index])
        index += 1
    return "".join(normalized)


def _validate_import_field(value: str) -> str:
    normalized = _normalize_unicode_scalar(value)
    if len(normalized.encode("utf-8")) > MAX_IMPORT_FIELD_UTF8_BYTES:
        raise ValueError(
            f"Launch-monitor field exceeds {MAX_IMPORT_FIELD_UTF8_BYTES} UTF-8 bytes"
        )
    return normalized


def _validate_csv_field_bytes(text: str) -> None:
    """Preflight decoded CSV fields without mutating csv.field_size_limit()."""
    byte_count = 0
    quoted = False
    index = 0
    while index < len(text):
        character = text[index]
        if character == '"':
            if quoted and index + 1 < len(text) and text[index + 1] == '"':
                byte_count += 1
                index += 2
            else:
                quoted = not quoted
                index += 1
        elif not quoted and character in {",", "\r", "\n"}:
            byte_count = 0
            index += 1
        else:
            byte_count += len(character.encode("utf-8"))
            index += 1
        if byte_count > MAX_IMPORT_FIELD_UTF8_BYTES:
            raise ValueError(
                "Launch-monitor field exceeds "
                f"{MAX_IMPORT_FIELD_UTF8_BYTES} UTF-8 bytes"
            )


def _validate_import_shape(row_count: int, column_count: int) -> None:
    if (
        isinstance(row_count, bool)
        or isinstance(column_count, bool)
        or not isinstance(row_count, int)
        or not isinstance(column_count, int)
        or row_count < 0
        or column_count < 0
    ):
        raise TypeError("launch-monitor import shape must use nonnegative integers")
    if row_count > MAX_IMPORT_ROWS:
        raise ValueError(f"Launch-monitor import exceeds {MAX_IMPORT_ROWS} rows")
    if column_count > MAX_IMPORT_COLUMNS:
        raise ValueError(f"Launch-monitor import exceeds {MAX_IMPORT_COLUMNS} columns")
    if row_count * column_count > MAX_IMPORT_CELLS:
        raise ValueError(
            f"Launch-monitor import exceeds {MAX_IMPORT_CELLS} dense cells"
        )


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
        normalized_key = _validate_import_field(key)
        normalized_value = (
            _validate_import_field(value) if isinstance(value, str) else value
        )
        if normalized_key in result:
            raise ValueError(
                f"Duplicate JSON field name is not permitted: {normalized_key}"
            )
        result[normalized_key] = normalized_value
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
        _validate_csv_field_bytes(text)
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
                if len(rows) > MAX_IMPORT_ROWS + 1:
                    raise ValueError(
                        f"Launch-monitor import exceeds {MAX_IMPORT_ROWS} rows"
                    )
        except csv.Error as error:
            raise ValueError("Launch-monitor CSV is malformed") from error
        if len(rows) < 2:
            raise ValueError("CSV must contain a header and at least one row")
        headers = tuple(item.strip() for item in rows[0])
        for value in headers:
            _validate_import_field(value)
        for row in rows[1:]:
            for value in row:
                _validate_import_field(value)
        if any(not item for item in headers) or len(set(headers)) != len(headers):
            raise ValueError("CSV headers must be non-empty and unique")
        if any(len(row) != len(headers) for row in rows[1:]):
            raise ValueError("Every CSV data row must match the header width")
        _validate_import_shape(len(rows) - 1, len(headers))
        # `_coerce_csv_cell` has already decided each cell's Python type, so let
        # pandas store them verbatim instead of re-inferring. pandas 3.0 infers
        # `StringDtype(na_value=nan)` for a text column, which turns the `None`
        # that marks a blank field into `nan` and breaks the browser-policy
        # contract that a blank field reads back as `null`/`None`.
        return pd.DataFrame(
            [[_coerce_csv_cell(value) for value in row] for row in rows[1:]],
            columns=headers,
            dtype=object,
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
        records = cast(list[dict[str, object]], payload)
        _validate_import_shape(len(records), 0)
        union: set[str] = set()
        for record in records:
            union.update(record)
            _validate_import_shape(0, len(union))
        _validate_import_shape(len(records), len(union))
        for record in records:
            for key, imported_value in record.items():
                _validate_import_field(key)
                if isinstance(imported_value, str):
                    _validate_import_field(imported_value)
        if any(not key.strip() for record in records for key in record):
            raise ValueError("JSON launch-monitor field names must be non-empty")
        if any(
            not _json_scalar_is_portable(value)
            for record in records
            for value in record.values()
        ):
            raise ValueError(
                "JSON launch-monitor record values must be portable finite scalars"
            )
        return pd.DataFrame.from_records(records)
    raise AssertionError("validated launch-monitor suffix was not dispatched")


__all__ = [
    "MAX_IMPORT_BYTES",
    "MAX_IMPORT_FIELD_UTF8_BYTES",
    "MAX_IMPORT_ROWS",
    "read_launch_monitor_frame",
]
