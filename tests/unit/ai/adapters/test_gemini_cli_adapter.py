"""Behavioral tests for GeminiCliAdapter (Tools #3178)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.shared.python.ai.adapters.cli_subprocess_runner import CliSubprocessResult
from src.shared.python.ai.adapters.gemini_cli_adapter import GeminiCliAdapter
from src.shared.python.ai.exceptions import (
    AIConnectionError,
    AIProviderError,
    AITimeoutError,
)
from src.shared.python.ai.types import ConversationContext

pytestmark = pytest.mark.unit

_RUN = "src.shared.python.ai.adapters.gemini_cli_adapter.invoke_cli_subprocess_for_adapter"


def _adapter(model: str | None = None) -> GeminiCliAdapter:
    adapter = GeminiCliAdapter(model=model)
    adapter.binary = "/usr/bin/gemini"
    return adapter


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> CliSubprocessResult:
    return CliSubprocessResult(returncode=returncode, stdout=stdout, stderr=stderr)


class TestSendMessage:
    def test_success_returns_agent_response(self) -> None:
        adapter = _adapter()
        with patch(_RUN, return_value=_completed(0, stdout="gemini reply\n")) as run:
            response = adapter.send_message("hi", ConversationContext(), [])
        assert response.content == "gemini reply"
        assert response.metadata["provider"] == "gemini_cli"
        args = run.call_args.args[1]
        assert "--skip-trust" in args
        assert "-p" in args

    def test_model_inserted_into_argv(self) -> None:
        adapter = _adapter(model="gemini-2.5-pro")
        with patch(_RUN, return_value=_completed(0, stdout="ok")) as run:
            adapter.send_message("hi", ConversationContext(), [])
        args = run.call_args.args[1]
        assert "--model" in args
        assert "gemini-2.5-pro" in args

    def test_empty_message_raises_value_error(self) -> None:
        adapter = _adapter()
        with pytest.raises(ValueError, match="non-empty"):
            adapter.send_message("  ", ConversationContext(), [])

    def test_missing_binary_raises_connection_error(self) -> None:
        adapter = GeminiCliAdapter()
        adapter.binary = None
        with pytest.raises(AIConnectionError):
            adapter.send_message("hi", ConversationContext(), [])

    def test_nonzero_exit_raises_provider_error(self) -> None:
        adapter = _adapter()
        with (
            patch(_RUN, return_value=_completed(4, stderr="quota exceeded")),
            pytest.raises(AIProviderError) as exc,
        ):
            adapter.send_message("hi", ConversationContext(), [])
        assert "quota exceeded" in str(exc.value)

    def test_timeout_raises_timeout_error(self) -> None:
        adapter = _adapter()
        with (
            patch(_RUN, side_effect=AITimeoutError("timed out", provider="gemini_cli")),
            pytest.raises(AITimeoutError),
        ):
            adapter.send_message("hi", ConversationContext(), [])

    def test_file_not_found_midrun_raises_connection_error(self) -> None:
        adapter = _adapter()
        with (
            patch(
                _RUN,
                side_effect=AIConnectionError("gone", provider="gemini_cli"),
            ),
            pytest.raises(AIConnectionError),
        ):
            adapter.send_message("hi", ConversationContext(), [])


class TestValidateConnection:
    def test_no_binary_reports_failure(self) -> None:
        adapter = GeminiCliAdapter()
        adapter.binary = None
        ok, msg = adapter.validate_connection()
        assert ok is False
        assert "not found" in msg

    def test_version_success(self) -> None:
        adapter = _adapter()
        with patch(_RUN, return_value=_completed(0, stdout="gemini 0.9\n")):
            ok, msg = adapter.validate_connection()
        assert ok is True
        assert "0.9" in msg

    def test_version_nonzero_exit_reports_failure(self) -> None:
        adapter = _adapter()
        with patch(_RUN, return_value=_completed(1, stderr="not authed")):
            ok, msg = adapter.validate_connection()
        assert ok is False
        assert "not authed" in msg

    def test_version_timeout_reports_failure(self) -> None:
        adapter = _adapter()
        with patch(_RUN, side_effect=AITimeoutError("slow", provider="gemini_cli")):
            ok, msg = adapter.validate_connection()
        assert ok is False
        assert "did not respond" in msg

    def test_version_oserror_reports_failure(self) -> None:
        adapter = _adapter()
        with patch(
            _RUN,
            side_effect=AIConnectionError("bad exec", provider="gemini_cli"),
        ):
            ok, msg = adapter.validate_connection()
        assert ok is False
        assert "bad exec" in msg
