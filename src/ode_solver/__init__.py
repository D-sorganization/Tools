# ODE Solver - Standalone GUI Package
"""ODE Solver GUI for symbolic ODE systems."""

from pathlib import Path

__version__ = "1.0.0"

_nested_package_dir = Path(__file__).resolve().parent / "python" / "ode_solver"
if _nested_package_dir.is_dir():
    __path__.append(str(_nested_package_dir))
