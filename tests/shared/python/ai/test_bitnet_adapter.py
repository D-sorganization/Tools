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
    ("src.shared.python.config", "src/shared/python/config"),
    ("src.shared.python.ai", "src/shared/python/ai"),
    ("src.shared.python.ai.adapters", "src/shared/python/ai/adapters"),
]
for _mod_name, _rel_path in _PACKAGE_STUBS:
    if _mod_name not in sys.modules:
        import types
        _stub = types.ModuleType(_mod_name)
        if _rel_path is not None:
            _stub.__path__ = [str(ROOT / _rel_path)]
        sys.modules[_mod_name] = _stub




# Stub logging_pkg so adapter modules can import get_logger.
_logging_config_stub = sys.modules.setdefault("src.shared.python.logging_pkg.logging_config", types.ModuleType("src.shared.python.logging_pkg.logging_config"))
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

    def test_stream_response_yields_chunks_for_each_line(
        self, adapter: BitnetAdapter
    ) -> None:
        """Each line of stdout becomes one AgentChunk, final chunk is empty."""
        mock_process = MagicMock()
        mock_process.stdout = iter(["Hello\n", " world\n", "!\n"])
        mock_process.wait.return_value = 0

        with patch("subprocess.Popen", return_value=mock_process):
            chunks = list(adapter.stream_response("hi", _make_context(), []))

        content_chunks = [c for c in chunks if not c.is_final]
        assert len(content_chunks) == 3
        assert chunks[-1].is_final is True
        assert chunks[-1].content == ""


# ---------------------------------------------------------------------------
# Constructor / initialization
# ---------------------------------------------------------------------------


class TestInit:
    """Tests for BitnetAdapter.__init__."""

    @pytest.mark.unit
    def test_default_model_name(self) -> None:
        """Default model is bitnet-1.58b-q4_0.gguf when none is provided."""
        adapter = BitnetAdapter()
        assert adapter.model == "bitnet-1.58b-q4_0.gguf"

    @pytest.mark.unit
    def test_custom_model_and_root(self) -> None:
        """Custom model and bitnet_root are stored verbatim."""
        adapter = BitnetAdapter(model="my-model.gguf", bitnet_root="/opt/bitnet")
        assert adapter.model == "my-model.gguf"
        assert adapter.bitnet_root == "/opt/bitnet"

    @pytest.mark.unit
    def test_llama_cli_path_uses_bitnet_root(self) -> None:
        """llama_cli path is joined from bitnet_root when root is set."""
        adapter = BitnetAdapter(bitnet_root="/opt/bitnet")
        assert adapter.llama_cli.startswith("/opt/bitnet")
        assert "llama-cli" in adapter.llama_cli

    @pytest.mark.unit
    def test_llama_cli_falls_back_to_plain_binary(self) -> None:
        """When bitnet_root is empty/unset, llama_cli defaults to 'llama-cli'."""
        adapter = BitnetAdapter(bitnet_root="")
        assert adapter.llama_cli == "llama-cli"

    @pytest.mark.unit
    def test_capabilities_provider_name_is_bitnet(self) -> None:
        """capabilities.provider_name is always 'bitnet'."""
        adapter = BitnetAdapter()
        assert adapter.capabilities.provider_name == "bitnet"

    @pytest.mark.unit
    def test_capabilities_model_name_matches_init(self) -> None:
        """capabilities.model_name reflects the model passed to __init__."""
        adapter = BitnetAdapter(model="custom.gguf")
        assert adapter.capabilities.model_name == "custom.gguf"


# ---------------------------------------------------------------------------
# validate_connection
# ---------------------------------------------------------------------------


class TestValidateConnection:
    """Tests for BitnetAdapter.validate_connection."""

    @pytest.mark.unit
    def test_returns_true_when_executable_responds(
        self, adapter: BitnetAdapter
    ) -> None:
        """validate_connection returns (True, msg) when llama-cli --help exits 0."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "usage: llama-cli"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            ok, msg = adapter.validate_connection()

        assert ok is True
        assert adapter.llama_cli in msg

    @pytest.mark.unit
    def test_returns_false_when_executable_missing(
        self, adapter: BitnetAdapter
    ) -> None:
        """validate_connection returns (False, msg) when binary is not found."""
        with patch("subprocess.run", side_effect=FileNotFoundError("no such file")):
            ok, msg = adapter.validate_connection()

        assert ok is False
        assert msg  # non-empty diagnostic

    @pytest.mark.unit
    def test_returns_true_when_usage_in_stdout(self, adapter: BitnetAdapter) -> None:
        """validate_connection returns True when 'usage' in stdout (non-zero exit)."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = "usage: llama-cli [options]"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            ok, _ = adapter.validate_connection()

        assert ok is True


# ---------------------------------------------------------------------------
# _format_prompt
# ---------------------------------------------------------------------------


class TestFormatPrompt:
    """Tests for BitnetAdapter._format_prompt."""

    @pytest.mark.unit
    def test_appends_assistant_suffix(self, adapter: BitnetAdapter) -> None:
        """Prompt always ends with 'Assistant:' ready for the model to continue."""

        ctx = _make_context()
        result = adapter._format_prompt(ctx, "What is 2+2?")
        assert result.endswith("Assistant:")

    @pytest.mark.unit
    def test_includes_user_message(self, adapter: BitnetAdapter) -> None:
        """The current user message appears in the formatted prompt."""
        ctx = _make_context()
        result = adapter._format_prompt(ctx, "unique-probe-msg")
        assert "unique-probe-msg" in result

    @pytest.mark.unit
    def test_empty_context_still_works(self, adapter: BitnetAdapter) -> None:
        """_format_prompt handles an empty context (no prior messages) gracefully."""
        ctx = ConversationContext(
            messages=[],
            user_expertise=ExpertiseLevel.INTERMEDIATE,
        )
        result = adapter._format_prompt(ctx, "hello")
        assert "hello" in result
        assert "Assistant:" in result


# ---------------------------------------------------------------------------
# send_message success path
# ---------------------------------------------------------------------------


class TestSendMessageSuccess:
    """Tests for the happy path of send_message."""

    @pytest.mark.unit
    def test_returns_agent_response_with_content(self, adapter: BitnetAdapter) -> None:
        """send_message wraps stdout output in an AgentResponse."""
        full_output = "User: hi\nAssistant: response text here"
        mock_result = MagicMock()
        mock_result.stdout = full_output
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            response = adapter.send_message("hi", _make_context(), [])

        from src.shared.python.ai.types import AgentResponse

        assert isinstance(response, AgentResponse)
        assert response.content  # non-empty after prompt strip

    @pytest.mark.unit
    def test_command_includes_model_flag(self, adapter: BitnetAdapter) -> None:
        """subprocess.run is called with '-m <model>' in the command."""
        mock_result = MagicMock()
        mock_result.stdout = "result"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            adapter.send_message("hi", _make_context(), [])

        call_args = mock_run.call_args[0][0]  # positional cmd list
        assert "-m" in call_args
        assert adapter.model in call_args
