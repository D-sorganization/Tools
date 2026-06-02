"""Behavioral tests for ClineAdapter (Tools #3178).

Unlike the CLI adapters, Cline speaks an OpenAI-compatible HTTP API, so these
tests mock the httpx client (``adapter._client``) rather than
``subprocess.run``. ``send_message`` routes all transport failures through
``BaseAgentAdapter._classify_error``, whose string-scan maps the error to the
correct ``AIError`` subclass:

- success → ``AgentResponse`` parsed from the OpenAI-shaped JSON,
- generic provider failure → ``AIProviderError``,
- connection failure → ``AIConnectionError``,
- timeout failure → ``AITimeoutError``,
- ``validate_connection`` happy / sad paths.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.shared.python.ai.adapters.cline_adapter import ClineAdapter
from src.shared.python.ai.exceptions import (
    AIConnectionError,
    AIProviderError,
    AITimeoutError,
)
from src.shared.python.ai.types import ConversationContext

pytestmark = pytest.mark.unit


def _adapter_with_client(client: MagicMock) -> ClineAdapter:
    adapter = ClineAdapter()
    adapter._client = client  # bypass lazy httpx construction
    return adapter


def _ok_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = payload
    return resp


class TestSendMessage:
    def test_success_parses_openai_response(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {"content": "cline says hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7},
            "model": "cline",
        }
        client = MagicMock()
        client.post.return_value = _ok_response(payload)
        adapter = _adapter_with_client(client)

        response = adapter.send_message("hi", ConversationContext(), [])
        assert response.content == "cline says hi"
        # Token counts normalized to canonical keys (issue #2763).
        assert response.usage["input_tokens"] == 5
        assert response.usage["output_tokens"] == 7

    def test_empty_message_violates_precondition(self) -> None:
        adapter = _adapter_with_client(MagicMock())
        # The @precondition decorator raises ContractViolationError, which
        # derives from ValueError (see shared.python.contracts).
        with pytest.raises(ValueError):
            adapter.send_message("   ", ConversationContext(), [])

    def test_generic_error_raises_provider_error(self) -> None:
        client = MagicMock()
        client.post.side_effect = ValueError("unexpected response shape")
        adapter = _adapter_with_client(client)
        with pytest.raises(AIProviderError):
            adapter.send_message("hi", ConversationContext(), [])

    def test_connection_error_classified(self) -> None:
        client = MagicMock()
        client.post.side_effect = RuntimeError("connection refused by server")
        adapter = _adapter_with_client(client)
        with pytest.raises(AIConnectionError):
            adapter.send_message("hi", ConversationContext(), [])

    def test_timeout_error_classified(self) -> None:
        client = MagicMock()
        client.post.side_effect = RuntimeError("request timed out")
        adapter = _adapter_with_client(client)
        with pytest.raises(AITimeoutError):
            adapter.send_message("hi", ConversationContext(), [])


class TestValidateConnection:
    def test_success_when_models_endpoint_ok(self) -> None:
        client = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        client.get.return_value = resp
        adapter = _adapter_with_client(client)

        ok, msg = adapter.validate_connection()
        assert ok is True
        assert "Connected" in msg

    def test_non_200_reports_failure(self) -> None:
        client = MagicMock()
        resp = MagicMock()
        resp.status_code = 503
        client.get.return_value = resp
        adapter = _adapter_with_client(client)

        ok, msg = adapter.validate_connection()
        assert ok is False
        assert "503" in msg

    def test_connection_error_reports_failure(self) -> None:
        client = MagicMock()
        client.get.side_effect = ConnectionError("refused")
        adapter = _adapter_with_client(client)

        ok, msg = adapter.validate_connection()
        assert ok is False
        assert "Cannot connect" in msg
