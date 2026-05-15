"""Unit tests for BitnetAdapter exception hierarchy.

Verifies that BitnetAdapter raises exceptions from the AIProviderError hierarchy
rather than bare RuntimeError, ensuring consistent error handling across all adapters.

The bootstrap block below follows the same pattern as test_adapter_factory.py:
it pre-stubs the broken src.shared.python.ai __init__ and logging_pkg so that
importing the adapter module directly works in a plain pytest run.
"""

from __future__ import annotations

import logging
import subprocess
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
_logging_config_stub.get_logger = logging.getLogger  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Now import the adapter and the exception hierarchy.
# ---------------------------------------------------------------------------

from src.shared.python.ai.adapters.bitnet_adapter import BitnetAdapter  # noqa: E402
from src.shared.python.ai.exceptions import (  # noqa: E402
    AIConnectionError,
    AIProviderError,
)
from src.shared.python.ai.types import ConversationContext, ExpertiseLevel  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context() -> ConversationContext:
    """Return a minimal ConversationContext for testing."""
    return ConversationContext(
        messages=[],
        user_expertise=ExpertiseLevel.INTERMEDIATE,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def adapter() -> BitnetAdapter:
    """Return a BitnetAdapter with a dummy binary path."""
    return BitnetAdapter(model="test-model.gguf", bitnet_root="/fake/root")


# ---------------------------------------------------------------------------
# send_message exception mapping
# ---------------------------------------------------------------------------


class TestSendMessageExceptions:
    """Tests for send_message exception mapping."""

    def test_called_process_error_raises_ai_provider_error(
        self, adapter: BitnetAdapter
    ) -> None:
        """CalledProcessError from subprocess.run maps to AIProviderError."""
        error = subprocess.CalledProcessError(
            returncode=1, cmd=["llama-cli"], stderr="model load failed"
        )
        with patch("subprocess.run", side_effect=error):
            with pytest.raises(AIProviderError) as exc_info:
                adapter.send_message("hello", _make_context(), [])

        assert "BitNet process failed" in str(exc_info.value)
        assert exc_info.value.provider == "bitnet"

    def test_called_process_error_is_subclass_of_ai_provider_error(
        self, adapter: BitnetAdapter
    ) -> None:
        """Raised exception from CalledProcessError IS-A AIProviderError (contract)."""
        error = subprocess.CalledProcessError(
            returncode=2, cmd=["llama-cli"], stderr="crash"
        )
        with patch("subprocess.run", side_effect=error):
            with pytest.raises(AIProviderError):
                adapter.send_message("hello", _make_context(), [])

    def test_file_not_found_raises_ai_connection_error(
        self, adapter: BitnetAdapter
    ) -> None:
        """FileNotFoundError (missing binary) maps to AIConnectionError."""
        fnf_exc = FileNotFoundError("llama-cli not found")
        with patch("subprocess.run", side_effect=fnf_exc):
            with pytest.raises(AIConnectionError) as exc_info:
                adapter.send_message("hello", _make_context(), [])

        assert "Failed to run BitNet" in str(exc_info.value)
        assert exc_info.value.provider == "bitnet"

    def test_file_not_found_is_subclass_of_ai_provider_error(
        self, adapter: BitnetAdapter
    ) -> None:
        """AIConnectionError IS-A AIProviderError (hierarchy contract)."""
        with patch("subprocess.run", side_effect=FileNotFoundError("no binary")):
            with pytest.raises(AIProviderError):
                adapter.send_message("hello", _make_context(), [])

    def test_no_bare_runtime_error_on_called_process_error(
        self, adapter: BitnetAdapter
    ) -> None:
        """Regression: bare RuntimeError must NOT be raised on CalledProcessError."""
        error = subprocess.CalledProcessError(
            returncode=1, cmd=["llama-cli"], stderr="err"
        )
        with patch("subprocess.run", side_effect=error):
            with pytest.raises(AIProviderError):
                adapter.send_message("hello", _make_context(), [])

    def test_no_bare_runtime_error_on_file_not_found(
        self, adapter: BitnetAdapter
    ) -> None:
        """Regression: bare RuntimeError must NOT be raised on FileNotFoundError."""
        with patch("subprocess.run", side_effect=FileNotFoundError("missing")):
            with pytest.raises(AIProviderError):
                adapter.send_message("hello", _make_context(), [])


# ---------------------------------------------------------------------------
# stream_response exception mapping
# ---------------------------------------------------------------------------


class TestStreamResponseExceptions:
    """Tests for stream_response exception mapping."""

    def test_stdout_unavailable_raises_ai_provider_error(
        self, adapter: BitnetAdapter
    ) -> None:
        """When Popen returns a process with no stdout, AIProviderError is raised."""
        mock_process = MagicMock()
        mock_process.stdout = None

        with patch("subprocess.Popen", return_value=mock_process):
            chunks = list(adapter.stream_response("hello", _make_context(), []))

        # stream_response catches AIProviderError and converts to a final error chunk
        assert any("Error" in chunk.content for chunk in chunks if chunk.is_final)

    @pytest.mark.parametrize(
        ("exc_class", "msg"),
        [
            (FileNotFoundError, "llama-cli"),
            (PermissionError, "permission denied"),
        ],
    )
    def test_popen_spawn_failure_yields_error_chunk(
        self,
        adapter: BitnetAdapter,
        exc_class: type[Exception],
        msg: str,
    ) -> None:
        """Popen failures are caught and surfaced as error chunks (not raised)."""
        with patch("subprocess.Popen", side_effect=exc_class(msg)):
            chunks = list(adapter.stream_response("hello", _make_context(), []))

        assert chunks[-1].is_final
