"""P1AM backend import contract (test-quality T-7, issue #5924).

Evidence class: installed-package / source-tree — flat ``main`` on backend path.
"""

from __future__ import annotations

import importlib
import sys

import pytest

from tests.p1am_control_system.backend_import import (
    _BACKEND_DIR,
    ensure_backend_on_path,
    load_backend_main,
)


def test_backend_main_path_points_at_backend_package() -> None:
    assert (_BACKEND_DIR / "main.py").is_file()


def test_load_backend_main_exposes_fastapi_app() -> None:
    main = load_backend_main()
    assert main.app is not None


def test_injected_main_import_failure_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("sqlmodel")
    ensure_backend_on_path()
    sys.modules.pop("main", None)

    original = importlib.import_module

    def _broken(name: str, *args: object, **kwargs: object):
        if name == "main":
            raise ImportError("injected backend import failure")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", _broken)
    with pytest.raises(ImportError, match="injected backend import failure"):
        load_backend_main()
