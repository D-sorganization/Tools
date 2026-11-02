"""Regression tests for the backend ``main`` import guard (#5924)."""

from __future__ import annotations

import importlib

import pytest
from p1am_app_import import import_main_app


def test_import_main_app_propagates_non_missing_import_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("sqlmodel")

    real_import_module = importlib.import_module

    def _broken_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "main":
            raise ImportError("simulated broken backend surface")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", _broken_import)
    with pytest.raises(ImportError, match="simulated broken backend surface"):
        import_main_app()
