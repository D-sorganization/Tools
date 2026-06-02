"""Behavioral tests for ClaudeCodeAdapter (Tools #3178).

The Claude Code adapter is a thin ``subprocess.run`` wrapper around the
``claude`` CLI. These tests mock ``subprocess.run`` so they run without the
binary, asserting:

- success → ``AgentResponse`` with the CLI stdout as content,
- non-zero exit → ``AIProviderError``,
- ``TimeoutExpired`` → ``AITimeoutError``,
- missing binary (``FileNotFoundError`` mid-run) → ``AIConnectionError``,
- unresolved binary → ``AIConnectionError``,
- ``validate_connection`` happy / sad paths.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.ai.adapters.claude_code_adapter import ClaudeCodeAdapter
from src.shared.python.ai.exceptions import (
    AIConnectionError,
    AIProviderError,
    AITimeoutError,
)
from src.shared.python.ai.types import ConversationContext

pytestmark = pytest.mark.unit

_RUN = "src.shared.python.ai.adapters.claude_code_adapter.subprocess.run"


def _adapter() -> ClaudeCodeAdapter:
    """Construct an adapter with a fixed binary so resolution is bypassed."""
    adapter = ClaudeCodeAdapter()
    adapter.binary = "/usr/bin/claude"
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
        with patch(_RUN, return_value=_completed(0, stdout="hello world\n")) as run:
            response = adapter.send_message("hi", ConversationContext(), [])
        assert response.content == "hello world"
        assert response.metadata["provider"] == "claude_code"
        run.assert_called_once()

    def test_empty_message_raises_value_error(self) -> None:
        adapter = _adapter()
        with pytest.raises(ValueError, match="non-empty"):
            adapter.send_message("   ", ConversationContext(), [])

    def test_missing_binary_raises_connection_error(self) -> None:
        adapter = ClaudeCodeAdapter()
        adapter.binary = None
        with pytest.raises(AIConnectionError):
            adapter.send_message("hi", ConversationContext(), [])

    def test_nonzero_exit_raises_provider_error(self) -> None:
        adapter = _adapter()
        with (
            patch(_RUN, return_value=_completed(2, stderr="boom")),
            pytest.raises(AIProviderError) as exc,
        ):
            adapter.send_message("hi", ConversationContext(), [])
        assert "boom" in str(exc.value)

    def test_timeout_raises_timeout_error(self) -> None:
        adapter = _adapter()
        with (
            patch(_RUN, side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=1)),
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


class TestValidateConnection:
    def test_no_binary_reports_failure(self) -> None:
        adapter = ClaudeCodeAdapter()
        adapter.binary = None
        ok, msg = adapter.validate_connection()
        assert ok is False
        assert "not found" in msg

    def test_version_success(self) -> None:
        adapter = _adapter()
        with patch(_RUN, return_value=_completed(0, stdout="claude 1.2.3\n")):
            ok, msg = adapter.validate_connection()
        assert ok is True
        assert "1.2.3" in msg

    def test_version_nonzero_exit_reports_failure(self) -> None:
        adapter = _adapter()
        with patch(_RUN, return_value=_completed(1, stderr="not logged in")):
            ok, msg = adapter.validate_connection()
        assert ok is False
        assert "not logged in" in msg

    def test_version_timeout_reports_failure(self) -> None:
        adapter = _adapter()
        with patch(
            _RUN, side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=10)
        ):
            ok, msg = adapter.validate_connection()
        assert ok is False
        assert "did not respond" in msg

    def test_version_oserror_reports_failure(self) -> None:
        adapter = _adapter()
        with patch(_RUN, side_effect=OSError("exec format error")):
            ok, msg = adapter.validate_connection()
        assert ok is False
        assert "Could not execute" in msg
