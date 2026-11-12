"""CalculatorSpec types for the calc_backend registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

Maturity = str


@dataclass(frozen=True)
class WorkedExample:
    """Minimal request/response pair for documentation and MCP describe."""

    request: dict[str, Any]
    response_summary: dict[str, Any] | None = None


@dataclass(frozen=True)
class FieldMetadata:
    """Per-field units and validation constraints for describe()."""

    name: str
    units: str | None
    valid_range: dict[str, Any]
    description: str | None


@dataclass(frozen=True)
class CalculatorSpec:
    """Single calculator schema source shared by REST and MCP."""

    id: str
    title: str
    category: str
    request_model: type[BaseModel]
    response_model: type[BaseModel] | None
    fn: Callable[..., Any]
    maturity: Maturity
    formula_ids: tuple[str, ...]
    units: Mapping[str, str]
    worked_example: WorkedExample
    route_subpath: str = ""
