"""Focused coverage for shared Python compatibility helpers."""

from __future__ import annotations

import enum
import importlib
import sys
from datetime import timezone
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


def test_python_310_fallback_exports_timezone_utc_and_str_enum() -> None:
    try:
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(sys, "version_info", (3, 10, 99, "final", 0))
            fallback_module = importlib.reload(compatibility)
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
    finally:
        importlib.reload(compatibility)
