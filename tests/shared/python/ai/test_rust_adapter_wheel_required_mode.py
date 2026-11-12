"""Unit tests for AI_BACKEND_WHEEL_REQUIRED strict import mode (#5684 slice 3)."""

from __future__ import annotations

import importlib
import sys

import pytest


def test_wheel_required_mode_fails_without_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Oracle 5: missing wheel must fail, not skip, when the gate env is set."""
    monkeypatch.setenv("AI_BACKEND_WHEEL_REQUIRED", "1")
    modules_to_drop = [
        k
        for k in sys.modules
        if k in {"ai_backend", "tests.shared.python.ai.test_rust_adapter_wheel_smoke"}
    ]
    for name in modules_to_drop:
        sys.modules.pop(name, None)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("tests.shared.python.ai.test_rust_adapter_wheel_smoke")
