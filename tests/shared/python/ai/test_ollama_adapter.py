import sys
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
    ("src.shared.python.config", "src/shared/python/config"),
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

# Mock ai.config so we don't hit the broken environment import
_config_stub = types.ModuleType("src.shared.python.ai.config")
_config_stub.get_ollama_host = MagicMock(return_value="http://localhost:11434")
_config_stub.get_ollama_model = MagicMock(return_value="llama3.1:8b")
_config_stub.get_ollama_timeout = MagicMock(return_value=120.0)
_config_stub.DEFAULT_OLLAMA_HOST = "http://localhost:11434"
_config_stub.DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
_config_stub.DEFAULT_OLLAMA_TIMEOUT = 120.0
sys.modules["src.shared.python.ai.config"] = _config_stub

from src.shared.python.ai.adapters.ollama_adapter import OllamaAdapter
from src.shared.python.ai.types import ConversationContext, ExpertiseLevel


@pytest.fixture
def adapter() -> OllamaAdapter:
    return OllamaAdapter(model="test-model")


@pytest.fixture
def context() -> ConversationContext:
    return ConversationContext(
        session_id="test-session",
        user_expertise=ExpertiseLevel.INTERMEDIATE,
    )


def test_ollama_adapter_format_messages_includes_response_style(adapter, context):
    """Verify that ResponseStyle is correctly injected into the system prompt."""
    context.response_style = "concise"

    with patch("httpx.Client") as mock_client:
        # We don't actually need to send the message, just check formatting
        messages = adapter._format_messages(context, "Hello", [])

        system_msg = next(m for m in messages if m["role"] == "system")
        assert "Reply concisely" in system_msg["content"]
        assert "Prefer code, tables" in system_msg["content"]


def test_ollama_adapter_format_messages_default_style(adapter, context):
    """Verify that default (standard) ResponseStyle is used."""
    # response_style defaults to "standard" in ConversationContext

    messages = adapter._format_messages(context, "Hello", [])

    system_msg = next(m for m in messages if m["role"] == "system")
    assert "Reply at a standard level of detail" in system_msg["content"]


def test_ollama_adapter_format_messages_detailed_style(adapter, context):
    """Verify that detailed ResponseStyle is used."""
    context.response_style = "detailed"

    messages = adapter._format_messages(context, "Hello", [])

    system_msg = next(m for m in messages if m["role"] == "system")
    assert "Reply in detail" in system_msg["content"]
    assert "Walk through reasoning" in system_msg["content"]
