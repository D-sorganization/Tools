"""calc_backend domain engines must import without a GUI toolkit (#3317).

The FastAPI calc_backend is a headless web service. Its domain engines used to
import the PyQt6 theme layer at module top, so importing ``WGSReactorEngine`` on
a server without PyQt6 failed — and the router then returned a misleading
"missing numpy/scipy" 503. These tests run a subprocess with PyQt6 import
*blocked* and assert the engine still imports and never pulls in the theme/Qt
layer.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Program run in a clean subprocess: block PyQt6 (and the theme layer that wraps
# it), then import the engine and assert success + no Qt/theme leakage.
_PROGRAM = textwrap.dedent("""
    import importlib.abc
    import importlib.machinery
    import sys


    class _BlockQt(importlib.abc.MetaPathFinder):
        _BLOCKED = ("PyQt6", "shared.python.theme.integration")

        def find_spec(self, fullname, path, target=None):
            if fullname == "PyQt6" or fullname.startswith("PyQt6."):
                raise ImportError("PyQt6 is blocked for this headless test")
            return None


    sys.meta_path.insert(0, _BlockQt())

    # Importing the engine must NOT require PyQt6 / the theme layer.
    from sidekick.process_calculators.wgs_reactor_calculator import WGSReactorEngine

    assert WGSReactorEngine is not None

    # A pure-thermodynamics call must work with no GUI present.
    engine = WGSReactorEngine()
    k = engine.calculate_equilibrium_constant(1000.0)
    assert k > 0

    # The GUI theme layer must not have been dragged in by the import.
    assert "shared.python.theme.integration" not in sys.modules, (
        "engine import pulled in the PyQt6 theme layer"
    )
    assert "PyQt6" not in sys.modules

    print("HEADLESS_OK")
    """)


@pytest.mark.unit
def test_wgs_engine_imports_without_pyqt6() -> None:
    # Reproduce the parent interpreter's import roots so the subprocess resolves
    # ``sidekick`` / ``contracts`` / ``shared`` the same way the test run does.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p and Path(p).is_dir())
    result = subprocess.run(
        [sys.executable, "-c", _PROGRAM],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert result.returncode == 0, (
        "headless engine import failed:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "HEADLESS_OK" in result.stdout
