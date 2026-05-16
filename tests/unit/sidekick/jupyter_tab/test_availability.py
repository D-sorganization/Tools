"""Tests for ``JupyterTabAvailability``."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest
from upstream_drift_tools.ui.tools_sidebar.jupyter_tab.availability import (
    JupyterTabAvailability,
)


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    JupyterTabAvailability.reset_cache()
    yield
    JupyterTabAvailability.reset_cache()


def test_check_returns_true_when_nbformat_is_importable() -> None:
    available, message = JupyterTabAvailability.check()
    assert available is True
    assert message == ""


def test_check_returns_install_hint_when_nbformat_missing() -> None:
    # Force a re-import attempt and make it fail to simulate missing dep.
    removed = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name == "nbformat" or name.startswith("nbformat.")
    }
    try:
        with patch.dict(sys.modules, {"nbformat": None}):
            available, message = JupyterTabAvailability.check()
    finally:
        sys.modules.update(removed)
    assert available is False
    assert "pip install" in message
    assert "jupyter" in message


def test_check_is_idempotent_and_cached() -> None:
    first = JupyterTabAvailability.check()
    second = JupyterTabAvailability.check()
    assert first == second
