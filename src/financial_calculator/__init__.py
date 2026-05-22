"""Package: financial_calculator."""

from __future__ import annotations

from pathlib import Path

_nested_package = Path(__file__).resolve().parent / "python" / "financial_calculator"
if _nested_package.is_dir():
    __path__.append(str(_nested_package))
