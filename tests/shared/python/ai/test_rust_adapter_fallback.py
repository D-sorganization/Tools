"""Tests verifying that the RustAgentAdapter emits a warning when the
ai_backend wheel is absent and the pure-Python path is activated.

The bootstrap block pre-stubs the src.shared.python.ai package hierarchy so
that importing the adapter module works in a plain ``pytest`` run (same pattern
as test_bitnet_adapter.py).
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: ensure repo root is on sys.path and stub broken package init
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_PACKAGE_STUBS: list[tuple[str, str | None]] = [
    ("src", "src"),
    ("src.shared", "src/shared"),
    ("src.shared.python", "src/shared/python"),
    ("src.shared.python.ai", "src/shared/python/ai"),
    ("src.shared.python.ai.adapters", "src/shared/python/ai/adapters"),
    ("src.shared.python.logging_pkg", None),
    ("src.shared.python.logging_pkg.logging_config", None),
]
for _mod_name, _rel_path in _PACKAGE_STUBS:
    if _mod_name not in sys.modules:
        _stub = types.ModuleType(_mod_name)
        if _rel_path is not None:
            _stub.__path__ = [str(ROOT / _rel_path)]  # type: ignore[attr-defined]
        sys.modules[_mod_name] = _stub

# Stub get_logger so adapter modules that call it don't break.
_logging_config_stub = sys.modules["src.shared.python.logging_pkg.logging_config"]
_logging_config_stub.get_logger = logging.getLogger  # type: ignore[attr-defined]

# Stub the base adapter and types so rust_adapter can be imported without the
# full AI package being installed.
_base_mod = types.ModuleType("src.shared.python.ai.adapters.base")


class _FakeBase:
    pass


_base_mod.BaseAgentAdapter = _FakeBase  # type: ignore[attr-defined]
_base_mod.ToolDeclaration = object  # type: ignore[attr-defined]
sys.modules["src.shared.python.ai.adapters.base"] = _base_mod

_types_mod = types.ModuleType("src.shared.python.ai.types")
for _name in (
    "AgentChunk",
    "AgentResponse",
    "ConversationContext",
    "ProviderCapabilities",
    "ProviderCapability",
):
    setattr(_types_mod, _name, object)
sys.modules["src.shared.python.ai.types"] = _types_mod

# Ensure the rust_adapter module is loaded (so its logger is bound to a real
# logger instance before we run any test).
sys.modules.pop("src.shared.python.ai.adapters.rust_adapter", None)
sys.modules.pop("ai_backend", None)

from src.shared.python.ai.adapters.rust_adapter import RustAgentAdapter  # noqa: E402

_ADAPTER_LOGGER = "src.shared.python.ai.adapters.rust_adapter"

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRustAdapterFallbackWarning:
    """RustAgentAdapter logs a WARNING when the ai_backend wheel is absent."""

    def test_warning_emitted_when_ai_backend_missing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When ai_backend is absent, a WARNING is logged before the ImportError."""
        with (
            patch.dict(sys.modules, {"ai_backend": None}),
            caplog.at_level(logging.WARNING, logger=_ADAPTER_LOGGER),
        ):
            with pytest.raises(ImportError):
                RustAgentAdapter(
                    api_key="test-key",
                    base_url="https://api.example.com/v1",
                    model="gpt-4",
                )

        warning_messages = [
            r.message for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert any(
            "ai_backend wheel not available" in str(msg) for msg in warning_messages
        ), f"Expected warning in logs, got: {warning_messages}"

    def test_warning_references_distribution_doc(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The warning message references docs/development/rust_distribution.md."""
        with (
            patch.dict(sys.modules, {"ai_backend": None}),
            caplog.at_level(logging.WARNING, logger=_ADAPTER_LOGGER),
        ):
            with pytest.raises(ImportError):
                RustAgentAdapter(
                    api_key="test-key",
                    base_url="https://api.example.com/v1",
                    model="gpt-4",
                )

        all_text = " ".join(str(r.message) for r in caplog.records)
        assert "rust_distribution.md" in all_text, (
            f"Expected 'rust_distribution.md' in log output, got: {all_text!r}"
        )
