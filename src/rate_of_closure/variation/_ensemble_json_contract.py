"""Bounded plain-JSON primitives for the Rate ensemble reader."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

import numpy as np

from shared.python.contracts import require

MAX_DECODED_NODES = 6_000_000
MAX_NESTING_DEPTH = 32
MAX_TEXT_CHARS = 4_096


def validate_decoded_tree(root: object) -> None:
    """Bound shape and scalar types before scientific materialization."""
    pending = [(root, 0)]
    nodes = 0
    while pending:
        value, depth = pending.pop()
        nodes += 1
        require(nodes <= MAX_DECODED_NODES, "decoded node limit exceeded", nodes)
        require(depth <= MAX_NESTING_DEPTH, "JSON nesting depth exceeded", depth)
        if isinstance(value, dict):
            require(
                all(isinstance(key, str) for key in value), "JSON keys must be strings"
            )
            pending.extend((child, depth + 1) for child in value.values())
        elif isinstance(value, list):
            pending.extend((child, depth + 1) for child in value)
        else:
            require(
                value is None or isinstance(value, (str, bool, int, float)),
                "JSON tree contains an unsupported value",
                type(value).__name__,
            )
            if isinstance(value, str):
                require(len(value) <= MAX_TEXT_CHARS, "JSON text limit exceeded")
            if isinstance(value, float):
                require(math.isfinite(value), "JSON numbers must be finite")


def numeric_matrix(
    value: object, rows: int, columns: int, nullable: bool
) -> np.ndarray:
    """Parse an exact rectangular numeric matrix."""
    row_values = json_list(value, "numeric matrix")
    require(len(row_values) == rows, "numeric matrix row count is invalid")
    parsed_rows = [json_list(raw_row, "numeric matrix row") for raw_row in row_values]
    require(
        all(len(row) == columns for row in parsed_rows),
        "numeric matrix column count is invalid",
    )
    result: np.ndarray = np.empty((rows, columns), dtype=float)
    for row_index, row in enumerate(parsed_rows):
        result[row_index] = [
            math.nan if item is None and nullable else number(item, "matrix value")
            for item in row
        ]
    return result


def number_vector(value: object, name: str) -> np.ndarray:
    """Parse one finite numeric vector."""
    return cast(
        np.ndarray,
        np.array([number(item, name) for item in json_list(value, name)], dtype=float),
    )


def bool_vector(value: object, length: int, name: str) -> np.ndarray:
    """Parse an exact-length boolean vector without integer coercion."""
    items = json_list(value, name)
    require(len(items) == length, f"{name} length is invalid")
    require(all(type(item) is bool for item in items), f"{name} must contain booleans")
    return cast(np.ndarray, np.array(items, dtype=bool))


def bool_matrix(value: object, rows: int, columns: int, name: str) -> np.ndarray:
    """Parse an exact rectangular boolean matrix."""
    items = json_list(value, name)
    require(len(items) == rows, f"{name} row count is invalid")
    parsed_rows = [json_list(raw_row, name) for raw_row in items]
    require(
        all(len(row) == columns for row in parsed_rows),
        f"{name} column count is invalid",
    )
    result: np.ndarray = np.empty((rows, columns), dtype=bool)
    for row_index, row in enumerate(parsed_rows):
        result[row_index] = bool_vector(row, columns, name)
    return result


def integer_vector(value: object, length: int, name: str) -> np.ndarray:
    """Parse an exact-length integer vector without boolean coercion."""
    items = json_list(value, name)
    require(len(items) == length, f"{name} length is invalid")
    return cast(
        np.ndarray,
        np.array([integer(item, name) for item in items], dtype=int),
    )


def string_tuple(value: object, name: str) -> tuple[str, ...]:
    """Parse an ordered JSON string array."""
    items = json_list(value, name)
    require(
        all(isinstance(item, str) for item in items), f"{name} must contain strings"
    )
    return tuple(items)


def mapping(value: object, name: str) -> dict[str, Any]:
    """Require a plain JSON object."""
    require(type(value) is dict, f"{name} must be an object", type(value).__name__)
    return cast(dict[str, Any], value)


def json_list(value: object, name: str) -> list[Any]:
    """Require a plain JSON array."""
    require(type(value) is list, f"{name} must be an array", type(value).__name__)
    return cast(list[Any], value)


def require_fields(data: Mapping[str, Any], expected: set[str], name: str) -> None:
    """Require an exact allowlisted object shape."""
    require(set(data) == expected, f"{name} must match the exact schema", tuple(data))


def number(value: object, name: str) -> float:
    """Require one finite JSON number, excluding booleans."""
    require(type(value) in (int, float), f"{name} must be a number", value)
    result = float(cast(int | float, value))
    require(math.isfinite(result), f"{name} must be finite", value)
    return result


def integer(value: object, name: str) -> int:
    """Require one JSON integer, excluding booleans."""
    require(type(value) is int, f"{name} must be an integer", value)
    return cast(int, value)


def exact_integer(value: object, name: str, expected: int) -> None:
    """Require one exact schema integer."""
    actual = integer(value, name)
    require(actual == expected, f"unsupported {name}", actual)


def optional_string(value: object, name: str) -> str | None:
    """Parse nullable bounded diagnostic text."""
    require(value is None or isinstance(value, str), f"{name} must be null or string")
    return cast(str | None, value)


__all__ = [
    "MAX_DECODED_NODES",
    "MAX_NESTING_DEPTH",
    "MAX_TEXT_CHARS",
    "bool_matrix",
    "bool_vector",
    "exact_integer",
    "integer",
    "integer_vector",
    "json_list",
    "mapping",
    "number",
    "number_vector",
    "numeric_matrix",
    "optional_string",
    "require_fields",
    "string_tuple",
    "validate_decoded_tree",
]
