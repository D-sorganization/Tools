"""Tests for P1AM backend import prerequisites (#5924)."""

from __future__ import annotations

import builtins
from types import ModuleType

import pytest

from tests.p1am_control_system.backend_import import require_backend_main


def test_require_backend_main_imports_when_backend_present() -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("sqlmodel")
    module = require_backend_main()
    assert isinstance(module, ModuleType)
    assert hasattr(module, "app")


def test_require_backend_main_propagates_unexpected_import_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("sqlmodel")
    original_import = builtins.__import__

    def _broken_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "main":
            raise ImportError("simulated backend defect")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _broken_import)
    with pytest.raises(ImportError, match="simulated backend defect"):
        require_backend_main()


def test_require_backend_main_skips_when_main_module_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("sqlmodel")
    original_import = builtins.__import__

    def _missing_main(name: str, *args: object, **kwargs: object) -> object:
        if name == "main":
            raise ModuleNotFoundError("main")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _missing_main)
    with pytest.raises(pytest.skip.Exception, match="not importable"):
        require_backend_main()
