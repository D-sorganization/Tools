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
    ("src.shared.python.config", "src/shared/python/config"),
    ("src.shared.python.ai", "src/shared/python/ai"),
    ("src.shared.python.ai.adapters", "src/shared/python/ai/adapters"),
]
for _mod_name, _rel_path in _PACKAGE_STUBS:
    if _mod_name not in sys.modules:
        import types

        _stub = types.ModuleType(_mod_name)
        if _rel_path is not None:
            _stub.__path__ = [str(ROOT / _rel_path)]
        sys.modules[_mod_name] = _stub


# Stub get_logger so adapter modules that call it don't break.
_logging_config_stub = sys.modules.setdefault(
    "src.shared.python.logging_pkg.logging_config",
    types.ModuleType("src.shared.python.logging_pkg.logging_config"),
)
_logging_config_stub.get_logger = logging.getLogger  # type: ignore[attr-defined]

# Stub ai.config so adapters can import without a real environment.
_env_stub = sys.modules.get("src.shared.python.config.environment")
if not isinstance(_env_stub, types.ModuleType):
    _env_stub = types.ModuleType("src.shared.python.config.environment")
    sys.modules["src.shared.python.config.environment"] = _env_stub
_env_stub.get_env = lambda key, default=None, required=False: default
_env_stub.get_env_float = lambda key, default=0.0: float(default)

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
        assert (
            "rust_distribution.md" in all_text
        ), f"Expected 'rust_distribution.md' in log output, got: {all_text!r}"
