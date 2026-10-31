"""Conftest for vessel drafter tests -- ensures package is importable."""

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

_PYTHON_DIR = str(Path(__file__).resolve().parent.parent / "python")
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip real-CAD tests when build123d is only a test stub (#5874).

    tests/vessel_drafter/conftest.py installs a MagicMock build123d so its
    tests can import the CAD modules; every xdist worker collects it, so
    ``importorskip("build123d")`` here returns the stub instead of skipping.
    """
    if item.get_closest_marker("requires_build123d") is None:
        return
    if isinstance(sys.modules.get("build123d"), Mock):
        pytest.skip("build123d is the tests/vessel_drafter stub, not the real kernel")
