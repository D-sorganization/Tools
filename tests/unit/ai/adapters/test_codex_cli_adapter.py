"""Behavioral tests for CodexCliAdapter (Tools #3178).

Mocks ``subprocess.run`` around the ``codex`` CLI wrapper, asserting the same
five branches as the other CLI adapters plus the ``_strip_telemetry`` helper
that drops Codex's bracketed timestamp preamble.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.ai.adapters.codex_cli_adapter import CodexCliAdapter
from src.shared.python.ai.exceptions import (
    AIConnectionError,
    AIProviderError,
    AITimeoutError,
)
from src.shared.python.ai.types import ConversationContext

pytestmark = pytest.mark.unit

_RUN = "src.shared.python.ai.adapters.codex_cli_adapter.subprocess.run"


def _adapter() -> CodexCliAdapter:
    adapter = CodexCliAdapter()
    adapter.binary = "/usr/bin/codex"
    return adapter


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> MagicMock:
    result = MagicMock()
    result.returncode = returncode
    result.stdout = stdout
    result.stderr = stderr
    return result


class TestSendMessage:
    def test_success_returns_agent_response(self) -> None:
        adapter = _adapter()
        with patch(_RUN, return_value=_completed(0, stdout="answer\n")) as run:
            response = adapter.send_message("hi", ConversationContext(), [])
        assert response.content == "answer"
        assert response.metadata["provider"] == "codex_cli"
        # The --skip-git-repo-check flag must always be present.
        args = run.call_args.args[0]
        assert "--skip-git-repo-check" in args

    def test_empty_message_raises_value_error(self) -> None:
        adapter = _adapter()
        with pytest.raises(ValueError, match="non-empty"):
            adapter.send_message("", ConversationContext(), [])

    def test_missing_binary_raises_connection_error(self) -> None:
        adapter = CodexCliAdapter()
        adapter.binary = None
        with pytest.raises(AIConnectionError):
            adapter.send_message("hi", ConversationContext(), [])

    def test_nonzero_exit_raises_provider_error(self) -> None:
        adapter = _adapter()
        with (
            patch(_RUN, return_value=_completed(3, stderr="auth failed")),
            pytest.raises(AIProviderError) as exc,
        ):
            adapter.send_message("hi", ConversationContext(), [])
        assert "auth failed" in str(exc.value)

    def test_timeout_raises_timeout_error(self) -> None:
        adapter = _adapter()
        with (
            patch(_RUN, side_effect=subprocess.TimeoutExpired(cmd="codex", timeout=1)),
            pytest.raises(AITimeoutError),
        ):
            adapter.send_message("hi", ConversationContext(), [])

    def test_file_not_found_midrun_raises_connection_error(self) -> None:
        adapter = _adapter()
        with (
            patch(_RUN, side_effect=FileNotFoundError("gone")),
            pytest.raises(AIConnectionError),
        ):
            adapter.send_message("hi", ConversationContext(), [])

    def test_telemetry_preamble_stripped_from_response(self) -> None:
        adapter = _adapter()
        stdout = "[2026-05-17T23:13:03] thinking…\nThe real answer.\n"
        with patch(_RUN, return_value=_completed(0, stdout=stdout)):
            response = adapter.send_message("hi", ConversationContext(), [])
        assert response.content == "The real answer."


class TestStripTelemetry:
    def test_drops_leading_timestamp_lines(self) -> None:
        raw = "[2026-01-01T00:00:00] boot\n[2026-01-01T00:00:01] warm\nHello"
        assert CodexCliAdapter._strip_telemetry(raw) == "Hello"

    def test_preserves_plain_output(self) -> None:
        assert CodexCliAdapter._strip_telemetry("just text") == "just text"

    def test_empty_string_returns_empty(self) -> None:
        assert CodexCliAdapter._strip_telemetry("") == ""


class TestValidateConnection:
    def test_no_binary_reports_failure(self) -> None:
        adapter = CodexCliAdapter()
        adapter.binary = None
        ok, msg = adapter.validate_connection()
        assert ok is False
        assert "not found" in msg

    def test_version_success(self) -> None:
        adapter = _adapter()
        with patch(_RUN, return_value=_completed(0, stdout="codex 0.4.0\n")):
            ok, msg = adapter.validate_connection()
        assert ok is True
        assert "0.4.0" in msg

    def test_version_nonzero_exit_reports_failure(self) -> None:
        adapter = _adapter()
        with patch(_RUN, return_value=_completed(1, stderr="broken")):
            ok, msg = adapter.validate_connection()
        assert ok is False
        assert "broken" in msg

    def test_version_timeout_reports_failure(self) -> None:
        adapter = _adapter()
        with patch(
            _RUN, side_effect=subprocess.TimeoutExpired(cmd="codex", timeout=10)
        ):
            ok, msg = adapter.validate_connection()
        assert ok is False
        assert "did not respond" in msg

    def test_version_oserror_reports_failure(self) -> None:
        adapter = _adapter()
        with patch(_RUN, side_effect=OSError("permission denied")):
            ok, msg = adapter.validate_connection()
        assert ok is False
        assert "Could not execute" in msg
