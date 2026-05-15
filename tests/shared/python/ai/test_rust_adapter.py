"""Tests for RustAgentAdapter non-blocking streaming."""

from __future__ import annotations

import sys
import threading
import time
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: stub the broken src.shared.python.ai __init__ and logging_pkg
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

# Stub logging_pkg so adapter modules can import get_logger.
_logging_config_stub = sys.modules["src.shared.python.logging_pkg.logging_config"]
_logging_config_stub.get_logger = MagicMock()  # type: ignore[attr-defined]

# Mock ai_backend before importing RustAgentAdapter
ai_backend_mock = MagicMock()
sys.modules["ai_backend"] = ai_backend_mock

# Mock PyQt6
pyqt6_mock = MagicMock()
sys.modules["PyQt6"] = pyqt6_mock
sys.modules["PyQt6.QtCore"] = pyqt6_mock.QtCore

from src.shared.python.ai.adapters.rust_adapter import RustAgentAdapter
from src.shared.python.ai.types import ConversationContext

def test_rust_adapter_stream_response_is_non_blocking_to_qt_events() -> None:
    """Verify that stream_response processes Qt events while waiting for Rust."""
    # Setup mocks
    mock_engine = MagicMock()
    mock_engine.stream_response.return_value = ["chunk1", "chunk2"]
    ai_backend_mock.AIEngine.return_value = mock_engine
    
    adapter = RustAgentAdapter(
        api_key="test",
        base_url="http://test",
        model="test-model"
    )
    
    context = ConversationContext(messages=[])
    
    # Setup mocks for PyQt environment
    mock_app = MagicMock()
    mock_thread = MagicMock()
    pyqt6_mock.QtCore.QCoreApplication.instance.return_value = mock_app
    pyqt6_mock.QtCore.QThread.currentThread.return_value = mock_thread
    mock_app.thread.return_value = mock_thread
    
    # We need to make stream_response take some time so we can check if processEvents was called
    def slow_stream(prompt):
        time.sleep(0.1)
        return ["chunk1", "chunk2"]
    
    mock_engine.stream_response.side_effect = slow_stream
    
    chunks = list(adapter.stream_response("test prompt", context, []))
    
    assert len(chunks) == 2
    assert chunks[0].content == "chunk1"
    assert chunks[1].content == "chunk2"
    
    # Verify processEvents was called at least once during wait
    assert mock_app.processEvents.called

def test_rust_adapter_stream_response_fallback_on_error() -> None:
    """Verify that stream_response falls back to generate_response on error."""
    mock_engine = MagicMock()
    mock_engine.stream_response.side_effect = Exception("stream failed")
    mock_engine.generate_response.return_value = "full response"
    ai_backend_mock.AIEngine.return_value = mock_engine
    
    adapter = RustAgentAdapter(
        api_key="test",
        base_url="http://test",
        model="test-model"
    )
    
    context = ConversationContext(messages=[])
    
    chunks = list(adapter.stream_response("test prompt", context, []))
    
    assert len(chunks) == 1
    assert chunks[0].content == "full response"
    assert chunks[0].is_final
    assert mock_engine.generate_response.called
