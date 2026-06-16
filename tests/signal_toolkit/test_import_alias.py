"""Compatibility tests for the signal-toolkit import shim."""

from __future__ import annotations

import importlib


def test_signal_toolkit_uses_shared_python_package() -> None:
    """The legacy top-level package resolves to the canonical shared package."""
    legacy = importlib.import_module("signal_toolkit")
    canonical = importlib.import_module("shared.python.signal_toolkit")

    assert legacy is canonical
