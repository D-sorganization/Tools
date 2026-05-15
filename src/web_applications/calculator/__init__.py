"""TI-89 inspired symbolic calculator utilities."""

from __future__ import annotations

from typing import Any

from .calculator import CalculatorResult, TI89Calculator

__all__ = ["CalculatorResult", "TI89Calculator", "create_app"]


def __getattr__(name: str) -> Any:
    """Lazy-load Flask app factory to avoid requiring Flask for headless imports."""
    if name == "create_app":
        from .webapp import create_app

        return create_app
    raise AttributeError(name)
