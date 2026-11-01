"""Focused coverage for shared Python compatibility helpers."""

from __future__ import annotations

import enum
import importlib
import importlib.util
import sys
from datetime import timezone
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

from src.shared.python import compatibility

pytestmark = pytest.mark.unit

TIMEZONE_UTC = timezone.utc  # noqa: UP017 - datetime.UTC is unavailable on Python 3.10.


class NativeLabel(compatibility.StrEnum):
    ALPHA = "alpha"


def test_native_python_aliases_use_standard_library_types() -> None:
    assert compatibility.UTC is TIMEZONE_UTC
    native_str_enum = getattr(enum, "StrEnum", None)
    if native_str_enum is None:
        assert compatibility.StrEnum is not enum.Enum
    else:
        assert compatibility.StrEnum is native_str_enum
    assert str(NativeLabel.ALPHA) == "alpha"
    assert cast(Any, NativeLabel.ALPHA).value == "alpha"


def _load_fresh_compatibility() -> ModuleType:
    """Execute ``compatibility.py`` into a new module object.

    ``importlib.reload`` is not enough: once an import-alias spelling such as
    the bare ``compatibility`` has been imported, the shared module's spec
    points at an alias loader whose ``exec_module`` never re-runs the source
    (#5902).
    """
    source = Path(cast(str, compatibility.__file__))
    spec = importlib.util.spec_from_file_location("_compatibility_under_test", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_python_310_fallback() -> None:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(sys, "version_info", (3, 10, 99, "final", 0))
        fallback_module = _load_fresh_compatibility()
    fallback_str_enum = cast(Any, fallback_module.StrEnum)

    fallback_label = fallback_str_enum("FallbackLabel", {"BETA": "beta"})

    assert fallback_module.UTC is TIMEZONE_UTC
    native_str_enum = getattr(enum, "StrEnum", None)
    if native_str_enum is not None:
        assert fallback_str_enum is not native_str_enum
    assert issubclass(fallback_str_enum, str)
    assert issubclass(fallback_str_enum, enum.Enum)
    assert str(fallback_label.BETA) == "beta"
    assert fallback_label.BETA.value == "beta"


def test_python_310_fallback_exports_timezone_utc_and_str_enum() -> None:
    _assert_python_310_fallback()


def test_python_310_fallback_holds_after_an_alias_import() -> None:
    """#5902: another test imports the bare ``compatibility`` alias first."""
    importlib.import_module("compatibility")
    _assert_python_310_fallback()
