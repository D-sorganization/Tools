"""CalculatorSpec registry for REST and MCP schema parity (issue #6020)."""

from .spec import CalculatorSpec, FieldMetadata, WorkedExample
from .store import (
    CALCULATOR_REGISTRY,
    bind_calculator_route,
    describe_calculator,
    http_path_for,
    register_from_catalog,
    registered_calculators,
)

__all__ = [
    "CALCULATOR_REGISTRY",
    "CalculatorSpec",
    "FieldMetadata",
    "WorkedExample",
    "bind_calculator_route",
    "describe_calculator",
    "http_path_for",
    "register_from_catalog",
    "registered_calculators",
]
