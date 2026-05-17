"""Test WebSocket error handling when AI providers (Ollama) fail."""

from __future__ import annotations

import json
from typing import AsyncIterator
from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.shared.python.ai.adapters.base import AIConnectionError
from src.shared.python.chat.models import AgentChunk
from src.shared.python.chat.router_factory import build_chat_router


@pytest.fixture
def mock_chat_service() -> Mock:
    service = Mock()
    # Return valid session for new session
    mock_session = Mock()
    mock_session.session_id = "test-session"
    service.get_or_create_session.return_value = mock_session
    return service


@pytest.fixture
def client(mock_chat_service: Mock) -> TestClient:
    app = FastAPI()
    app.state.chat_service = mock_chat_service
    router = build_chat_router(mock_chat_service)
    app.include_router(router)
    return TestClient(app)


def test_websocket_propagates_connection_error_without_disconnecting(
    client: TestClient, mock_chat_service: Mock
) -> None:
    """Test that an AIConnectionError is passed to the client as an error message and does not disconnect."""
    
    # Mock stream_response to raise an exception
    async def mock_stream(*args, **kwargs) -> AsyncIterator[AgentChunk]:
        yield AgentChunk(content="Thinking...", is_final=False, index=0)
        raise AIConnectionError("Cannot connect to Ollama", provider="ollama")
        
    mock_chat_service.stream_response.side_effect = mock_stream
    
    with client.websocket_connect("/ws/chat/test-session") as websocket:
        # Initial connect message
        data = websocket.receive_json()
        assert data["type"] == "session_info"
        assert data["session_id"] == "test-session"
        
        # Send a message
        websocket.send_json({"action": "send", "message": "Hello", "app_context": "tests"})
        
        # Should get chunk
        data = websocket.receive_json()
        assert data["type"] == "chunk"
        assert data["content"] == "Thinking..."
        
        # Should get error message, NOT a disconnect exception
        data = websocket.receive_json()
        assert data["type"] == "error"
        assert "Cannot connect to Ollama" in data["detail"]
        
        # Verify websocket is still open by sending history request
        websocket.send_json({"action": "history"})
        mock_chat_service.get_session_history.return_value = []
        data = websocket.receive_json()
        assert data["type"] == "history"
