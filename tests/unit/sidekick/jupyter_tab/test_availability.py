"""Tests for ``JupyterTabAvailability``."""

from __future__ import annotations

import builtins
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
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "nbformat" or name.startswith("nbformat."):
            raise ImportError("No module named 'nbformat'")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        available, message = JupyterTabAvailability.check()
    assert available is False
    assert "pip install" in message
    assert "jupyter" in message


def test_check_is_idempotent_and_cached() -> None:
    first = JupyterTabAvailability.check()
    second = JupyterTabAvailability.check()
    assert first == second
