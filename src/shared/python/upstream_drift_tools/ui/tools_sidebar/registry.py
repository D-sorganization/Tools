"""Workspace variable registry for sidebar-aware host applications."""

from __future__ import annotations

import builtins
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass(frozen=True)
class WorkspaceVariable:
    """Metadata snapshot for one workspace variable."""

    name: str
    value: Any
    type_name: str
    summary: str
    json_safe: bool
    repr_value: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-safe metadata for UI lists and persisted state."""
        data: dict[str, Any] = {
            "name": self.name,
            "type": self.type_name,
            "summary": self.summary,
            "json_safe": self.json_safe,
        }
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
        if initial:
            for name, value in initial.items():
                self.set(name, value)

    def set(self, name: str, value: Any) -> WorkspaceVariable:
        """Set a workspace variable and return its metadata snapshot."""
        self._validate_name(name)
        self._values[name] = value
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
        return existed

    def clear(self) -> None:
        """Remove all variables."""
        self._values.clear()
        self._repr_values.clear()

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
        return WorkspaceVariable(
            name=name,
            value=value,
            type_name=type(value).__name__,
            summary=_summarize_dimensions(value),
            json_safe=json_safe,
            repr_value=None if json_safe else self._repr_values.get(name, repr(value)),
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


def _summarize_dimensions(value: Any) -> str:
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return "shape=" + "x".join(str(part) for part in shape)
        except TypeError:
            return "shape=unknown"

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


def _env_key(name: str) -> str:
    return "".join(char.upper() if char.isalnum() else "_" for char in name)
