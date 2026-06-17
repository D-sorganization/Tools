"""Tests verifying that the RustAgentAdapter emits a warning when the
ai_backend wheel is absent and the pure-Python path is activated.

The bootstrap block pre-stubs the src.shared.python.ai package hierarchy so
that importing the adapter module works in a plain ``pytest`` run (same pattern
as test_bitnet_adapter.py).
"""

from __future__ import annotations

import logging
import sys
from unittest.mock import patch

import pytest

from src.shared.python.ai.adapters.rust_adapter import (  # noqa: E402
    RustAgentAdapter,
    RustBackendUnavailableError,
    ai_backend_available,
)

_ADAPTER_LOGGER = "src.shared.python.ai.adapters.rust_adapter"

_ADAPTER_KWARGS = {
    "api_key": "test-key",  # pragma: allowlist secret
    "base_url": "https://api.example.com/v1",
    "model": "gpt-4",
}

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


class TestGracefulDegradation:
    """The adapter cleanly reports unavailable instead of crashing consumers.

    Per issue #3521: a missing ``ai_backend`` wheel must NOT crash callers.
    ``try_create`` / ``is_available`` give a typed, exception-free path; when
    the wheel IS present behaviour is identical to direct construction.
    """

    def test_is_available_false_when_missing(self) -> None:
        """``ai_backend_available`` is False when the module cannot be imported."""
        with patch.dict(sys.modules, {"ai_backend": None}):
            assert ai_backend_available() is False
            assert RustAgentAdapter.is_available() is False

    def test_try_create_returns_none_when_missing(self) -> None:
        """``try_create`` degrades to ``None`` (no raise) when the wheel is absent."""
        with patch.dict(sys.modules, {"ai_backend": None}):
            adapter = RustAgentAdapter.try_create(**_ADAPTER_KWARGS)
        assert adapter is None

    def test_missing_wheel_raises_typed_error(self) -> None:
        """Direct construction raises the typed error (an ImportError subclass)."""
        with patch.dict(sys.modules, {"ai_backend": None}):
            with pytest.raises(RustBackendUnavailableError):
                RustAgentAdapter(**_ADAPTER_KWARGS)
        # Backwards-compat: existing `except ImportError` consumers still catch it.
        assert issubclass(RustBackendUnavailableError, ImportError)

    def test_uses_rust_when_present(self) -> None:
        """When a stub ``ai_backend`` is present, construction succeeds (Rust path).

        Proves the present-wheel behaviour is unchanged: ``try_create`` returns
        a real adapter and ``is_available`` reports True.
        """
        from unittest.mock import MagicMock

        fake_backend = MagicMock(name="ai_backend")
        with patch.dict(sys.modules, {"ai_backend": fake_backend}):
            assert ai_backend_available() is True
            adapter = RustAgentAdapter.try_create(**_ADAPTER_KWARGS)
            assert isinstance(adapter, RustAgentAdapter)
        # The Rust factory functions were exercised (used Rust, not a fallback).
        fake_backend.AIConfig.assert_called_once()
        fake_backend.AIEngine.assert_called_once()
