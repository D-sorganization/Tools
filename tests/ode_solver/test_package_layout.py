"""Package-layout regression tests for the ODE solver package."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def test_src_package_exposes_nested_ode_solver_modules(monkeypatch) -> None:
    """Importing ``ode_solver`` from ``src`` still exposes nested modules."""
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    nested_package_dir = src_root / "ode_solver" / "python" / "ode_solver"

    monkeypatch.syspath_prepend(str(src_root))
    for module_name in list(sys.modules):
        if module_name == "ode_solver" or module_name.startswith("ode_solver."):
            sys.modules.pop(module_name, None)

    package = importlib.import_module("ode_solver")
    package_paths = {str(Path(path)) for path in package.__path__}

    assert str(nested_package_dir) in package_paths
    assert importlib.import_module("ode_solver.timeout").SolverTimeoutError
