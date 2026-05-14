"""Workspace variable registry for sidebar-aware host applications."""

from __future__ import annotations

import builtins
import contextlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
PREVIEW_MAX_ROWS = 3
PREVIEW_MAX_COLUMNS = 4
PREVIEW_MAX_CHARS = 120


@dataclass(frozen=True)
class WorkspaceVariable:
    """Metadata snapshot for one workspace variable."""

    name: str
    value: Any
    type_name: str
    summary: str
    json_safe: bool
    repr_value: str | None = None
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    size: int | None = None
    preview: str = ""

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-safe metadata for UI lists and persisted state."""
        data: dict[str, Any] = {
            "name": self.name,
            "type": self.type_name,
            "summary": self.summary,
            "json_safe": self.json_safe,
            "preview": self.preview,
        }
        if self.shape is not None:
            data["shape"] = list(self.shape)
        if self.dtype is not None:
            data["dtype"] = self.dtype
        if self.size is not None:
            data["size"] = self.size
        if self.json_safe:
            data["value"] = self.value
        else:
            data["repr"] = self.repr_value or repr(self.value)
        return data


class WorkspaceRegistry:
    """Small in-memory registry for variables shared by tools and terminals.

    Values are intentionally untyped at runtime so host applications can register
    domain objects. Persistence keeps JSON-native values losslessly and stores
    non-JSON values as representation metadata instead of failing.
    """

    def __init__(self, initial: dict[str, Any] | None = None) -> None:
        self._values: dict[str, Any] = {}
        self._repr_values: dict[str, str] = {}
        self._metadata_overrides: dict[str, dict[str, Any]] = {}
        if initial:
            for name, value in initial.items():
                self.set(name, value)

    def set(self, name: str, value: Any) -> WorkspaceVariable:
        """Set a workspace variable and return its metadata snapshot."""
        self._validate_name(name)
        _validate_supported_value(value)
        self._values[name] = value
        self._metadata_overrides.pop(name, None)
        if _is_json_safe(value):
            self._repr_values.pop(name, None)
        else:
            self._repr_values[name] = repr(value)
        return self.describe(name)

    def get(self, name: str, default: Any = None) -> Any:
        """Return a variable value or ``default`` when absent."""
        return self._values.get(name, default)

    def remove(self, name: str) -> bool:
        """Remove a variable. Returns ``True`` when it existed."""
        existed = name in self._values
        self._values.pop(name, None)
        self._repr_values.pop(name, None)
        self._metadata_overrides.pop(name, None)
        return existed

    def clear(self) -> None:
        """Remove all variables."""
        self._values.clear()
        self._repr_values.clear()
        self._metadata_overrides.clear()

    def list(self) -> builtins.list[str]:
        """Return registered variable names in stable sorted order."""
        return sorted(self._values)

    def list_names(self) -> builtins.list[str]:
        """Alias for callers that avoid shadowing the built-in ``list``."""
        return self.list()

    def describe(self, name: str) -> WorkspaceVariable:
        """Return a metadata snapshot for one variable."""
        if name not in self._values:
            raise KeyError(name)
        value = self._values[name]
        json_safe = name not in self._repr_values and _is_json_safe(value)
        metadata = self._metadata_overrides.get(name) or _array_metadata(value)
        return WorkspaceVariable(
            name=name,
            value=value,
            type_name=type(value).__name__,
            summary=_summarize_dimensions(value, metadata),
            json_safe=json_safe,
            repr_value=None if json_safe else self._repr_values.get(name, repr(value)),
            shape=metadata["shape"],
            dtype=metadata["dtype"],
            size=metadata["size"],
            preview=metadata.get("preview") or format_workspace_value_preview(value),
        )

    def variables(self) -> builtins.list[WorkspaceVariable]:
        """Return metadata snapshots for all variables."""
        return [self.describe(name) for name in self.list()]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe payload suitable for persistence."""
        return {
            "version": 1,
            "variables": [variable.to_metadata() for variable in self.variables()],
        }

    def save_json(self, path: str | Path) -> None:
        """Persist registry metadata and JSON-safe values to ``path``."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> WorkspaceRegistry:
        """Load a registry saved by :meth:`save_json`."""
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        registry = cls()
        for entry in payload.get("variables", []):
            name = str(entry["name"])
            if entry.get("json_safe", False):
                registry.set(name, entry.get("value"))
            else:
                repr_value = str(entry.get("repr", ""))
                registry._values[name] = repr_value
                registry._repr_values[name] = repr_value
                registry._metadata_overrides[name] = _metadata_from_entry(entry)
        return registry

    def export_environment(self, prefix: str = "UD_VAR_") -> dict[str, str]:
        """Return stringified variables for terminal/process environments."""
        env: dict[str, str] = {}
        for name, variable in ((name, self.describe(name)) for name in self.list()):
            key = f"{prefix}{_env_key(name)}"
            if variable.json_safe:
                env[key] = json.dumps(variable.value)
            else:
                env[key] = variable.repr_value or repr(variable.value)
        return env

    @staticmethod
    def _validate_name(name: str) -> None:
        if not name or not name.strip():
            raise ValueError("Workspace variable name must be non-empty")


def _is_json_safe(value: Any) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return isinstance(value, str | int | float | bool | type(None) | list | dict)


def _validate_supported_value(value: Any) -> None:
    if _is_ragged_matrix(value):
        raise ValueError("ragged matrix values are not supported")


def _summarize_dimensions(value: Any, metadata: dict[str, Any]) -> str:
    shape = metadata["shape"]
    if shape is not None:
        return "x".join(str(part) for part in shape)

    if isinstance(value, dict):
        return f"keys={len(value)}"
    if isinstance(value, str):
        return f"length={len(value)}"
    if isinstance(value, list | tuple):
        if value and all(isinstance(row, list | tuple) for row in value):
            row_lengths = {len(row) for row in value}
            if len(row_lengths) == 1:
                return f"{len(value)}x{row_lengths.pop()}"
        return f"length={len(value)}"
    return "scalar"


def format_workspace_value_preview(value: Any) -> str:
    """Return a bounded MATLAB-like value preview for workspace UIs."""
    if isinstance(value, str):
        return _clip_preview(value)
    preview_value = _preview_source(value)
    return _clip_preview(repr(preview_value))


def _array_metadata(value: Any) -> dict[str, Any]:
    shape = _shape_of(value)
    return {
        "shape": shape,
        "dtype": _dtype_of(value),
        "size": _size_of(shape, value),
        "preview": format_workspace_value_preview(value),
    }


def _metadata_from_entry(entry: dict[str, Any]) -> dict[str, Any]:
    raw_shape = entry.get("shape")
    return {
        "shape": tuple(raw_shape) if isinstance(raw_shape, list) else None,
        "dtype": str(entry["dtype"]) if "dtype" in entry else None,
        "size": int(entry["size"]) if "size" in entry else None,
        "preview": str(entry.get("preview", entry.get("repr", ""))),
    }


def _shape_of(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is not None:
        with contextlib.suppress(TypeError, ValueError):
            return tuple(int(part) for part in shape)

    if isinstance(value, list | tuple):
        return _sequence_shape(value)
    return None


def _dtype_of(value: Any) -> str | None:
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        return str(dtype)
    if _shape_of(value) is None:
        return type(value).__name__
    scalar_types = {_scalar_type_name(item) for item in _flatten_sequence(value)}
    scalar_types.discard("NoneType")
    if not scalar_types:
        return "empty"
    if scalar_types <= {"int"}:
        return "int"
    if scalar_types <= {"int", "float"}:
        return "float"
    if scalar_types <= {"bool"}:
        return "bool"
    if len(scalar_types) == 1:
        return scalar_types.pop()
    return "object"


def _size_of(shape: tuple[int, ...] | None, value: Any) -> int | None:
    if shape is not None:
        return math.prod(shape)
    try:
        return int(len(value))
    except TypeError:
        return None


def _preview_source(value: Any) -> Any:
    array_values = _to_nested_lists(value)
    if array_values is not None:
        return _bounded_sequence(array_values)
    if isinstance(value, dict):
        items = list(value.items())[:PREVIEW_MAX_COLUMNS]
        result = {key: item_value for key, item_value in items}
        if len(value) > PREVIEW_MAX_COLUMNS:
            result["..."] = "..."
        return result
    return value


def _to_nested_lists(value: Any) -> Any | None:
    if isinstance(value, list | tuple):
        return _listify(value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    if hasattr(value, "shape") and hasattr(value, "__iter__"):
        try:
            return _listify(value)
        except TypeError:
            return None
    return None


def _bounded_sequence(value: Any, depth: int = 0) -> Any:
    if not isinstance(value, list):
        return value
    limit = PREVIEW_MAX_ROWS if depth == 0 else PREVIEW_MAX_COLUMNS
    result = [_bounded_sequence(item, depth + 1) for item in value[:limit]]
    if len(value) > limit:
        result.append("...")
    return result


def _sequence_shape(value: list[Any] | tuple[Any, ...]) -> tuple[int, ...]:
    if not value:
        return (0,)
    if all(isinstance(row, list | tuple) for row in value):
        row_shapes = {_sequence_shape(row) for row in value}
        if len(row_shapes) == 1:
            return (len(value), *row_shapes.pop())
        return (len(value),)
    return (len(value),)


def _is_ragged_matrix(value: Any) -> bool:
    if not isinstance(value, list | tuple) or not value:
        return False
    if not all(isinstance(row, list | tuple) for row in value):
        return False
    return len({_sequence_shape(row) for row in value}) > 1


def _flatten_sequence(value: Any) -> builtins.list[Any]:
    if isinstance(value, list | tuple):
        flattened: builtins.list[Any] = []
        for item in value:
            flattened.extend(_flatten_sequence(item))
        return flattened
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _flatten_sequence(tolist())
    return [value]


def _listify(value: Any) -> Any:
    if isinstance(value, list):
        return [_listify(item) for item in value]
    if isinstance(value, tuple):
        return [_listify(item) for item in value]
    if hasattr(value, "tolist"):
        return _listify(value.tolist())
    return value


def _scalar_type_name(value: Any) -> str:
    return type(value).__name__


def _clip_preview(preview: str) -> str:
    if len(preview) > PREVIEW_MAX_CHARS:
        return preview[: PREVIEW_MAX_CHARS - 3].rstrip() + "..."
    return preview


def _env_key(name: str) -> str:
    return "".join(char.upper() if char.isalnum() else "_" for char in name)
