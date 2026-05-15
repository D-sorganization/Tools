"""TI-89 inspired symbolic calculator utilities."""

from importlib import import_module
from typing import Any, cast

from .calculator import CalculatorResult, TI89Calculator

__all__ = ["CalculatorResult", "TI89Calculator", "create_app"]


def __getattr__(name: str) -> Any:
    if name == "create_app":
        webapp = cast(Any, import_module(".webapp", __name__))
        return webapp.create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
