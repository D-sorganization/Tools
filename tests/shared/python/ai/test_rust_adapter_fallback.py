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
