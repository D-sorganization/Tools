"""Unit tests for BitnetAdapter exception hierarchy.

Verifies that BitnetAdapter raises exceptions from the AIProviderError hierarchy
rather than bare RuntimeError, ensuring consistent error handling across all adapters.

The bootstrap block below follows the same pattern as test_adapter_factory.py:
it pre-stubs the broken src.shared.python.ai __init__ and logging_pkg so that
importing the adapter module directly works in a plain pytest run.
"""

from __future__ import annotations

import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Now import the adapter and the exception hierarchy.
# ---------------------------------------------------------------------------
from src.shared.python.ai.adapters.bitnet_adapter import BitnetAdapter  # noqa: E402
from src.shared.python.ai.exceptions import (  # noqa: E402
    AIConnectionError,
    AIProviderError,
    AITimeoutError,
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


# ---------------------------------------------------------------------------
# Timeout enforcement (issue #3175)
# ---------------------------------------------------------------------------


class TestTimeouts:
    """A hung llama-cli must be bounded by a wall-clock deadline."""

    @pytest.mark.unit
    def test_send_message_passes_timeout_to_subprocess(
        self, adapter: BitnetAdapter
    ) -> None:
        """send_message forwards self.timeout to subprocess.run."""
        mock_result = MagicMock()
        mock_result.stdout = "ok"
        mock_result.stderr = ""
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result) as run:
            adapter.send_message("hi", _make_context(), [])
        assert run.call_args.kwargs["timeout"] == adapter.timeout

    @pytest.mark.unit
    def test_send_message_timeout_raises_ai_timeout_error(
        self, adapter: BitnetAdapter
    ) -> None:
        """subprocess.TimeoutExpired maps to AITimeoutError (issue #3175)."""
        exc = subprocess.TimeoutExpired(cmd=["llama-cli"], timeout=adapter.timeout)
        with patch("subprocess.run", side_effect=exc):
            with pytest.raises(AITimeoutError) as info:
                adapter.send_message("hi", _make_context(), [])
        assert info.value.provider == "bitnet"

    @pytest.mark.unit
    def test_ai_timeout_error_is_ai_provider_error(
        self, adapter: BitnetAdapter
    ) -> None:
        """AITimeoutError IS-A AIProviderError (hierarchy contract)."""
        exc = subprocess.TimeoutExpired(cmd=["llama-cli"], timeout=adapter.timeout)
        with patch("subprocess.run", side_effect=exc):
            with pytest.raises(AIProviderError):
                adapter.send_message("hi", _make_context(), [])

    @pytest.mark.unit
    def test_constructor_rejects_nonpositive_timeout(self) -> None:
        """timeout <= 0 is a precondition violation (DbC)."""
        with pytest.raises(ValueError):
            BitnetAdapter(timeout=0)
        with pytest.raises(ValueError):
            BitnetAdapter(timeout=-1)

    @pytest.mark.unit
    def test_stream_terminates_hung_process_within_timeout(self) -> None:
        """A stub process that emits a banner then never EOFs is killed.

        The fake stdout yields one banner line, then blocks forever on the
        next ``__next__`` (simulating a CLI that never reaches EOF). With a
        tiny timeout the stream must terminate, kill the child, and yield a
        final error chunk — all well within a few seconds.
        """
        adapter = BitnetAdapter(
            model="test-model.gguf", bitnet_root="/fake/root", timeout=0.3
        )

        import threading

        never_eof = threading.Event()  # never set -> blocks forever

        class _HangingStdout:
            def __init__(self) -> None:
                self._emitted = False

            def __iter__(self) -> _HangingStdout:
                return self

            def __next__(self) -> str:
                if not self._emitted:
                    self._emitted = True
                    return "banner: loading model\n"
                # Block until the (never-set) event fires.
                never_eof.wait()
                raise StopIteration

        mock_process = MagicMock()
        mock_process.stdout = _HangingStdout()
        mock_process.poll.return_value = None  # appears alive
        mock_process.wait.return_value = 0

        start = time.monotonic()
        with patch("subprocess.Popen", return_value=mock_process):
            chunks = list(adapter.stream_response("hi", _make_context(), []))
        elapsed = time.monotonic() - start

        # Unblock the reader thread so it can exit cleanly post-assert.
        never_eof.set()

        assert elapsed < 5.0, "stream did not honor the deadline"
        assert chunks[-1].is_final is True
        assert "timed out" in chunks[-1].content.lower()
        # The hung child must have been killed/reaped.
        mock_process.kill.assert_called_once()


class TestPromptValidation:
    """The prompt is a single argv element, so it is bounded and encodable.

    `llama-cli` receives the whole conversation as one `-p` argument. Without
    these guards an unbounded prompt risks E2BIG or a silently truncated
    argument list, and text carrying a lone surrogate (reachable from a chat
    surface fed an undecodable upstream byte string) fails deep inside
    `subprocess` after the fork, with nothing tying the error back to the
    prompt. Both checks run before any process is spawned.
    """

    def test_send_message_rejects_oversize_prompt(self, adapter) -> None:
        message = "x" * (adapter._MAX_PROMPT_BYTES + 1)

        with (
            patch(
                "src.shared.python.ai.adapters.bitnet_adapter.subprocess.run"
            ) as mock_run,
            pytest.raises(AIProviderError, match="maximum size"),
        ):
            adapter.send_message(message, ConversationContext(), [])

        mock_run.assert_not_called()

    def test_send_message_rejects_invalid_utf8_prompt(self, adapter) -> None:
        with (
            patch(
                "src.shared.python.ai.adapters.bitnet_adapter.subprocess.run"
            ) as mock_run,
            pytest.raises(AIProviderError, match="valid UTF-8 text"),
        ):
            adapter.send_message("bad\udcff", ConversationContext(), [])

        mock_run.assert_not_called()

    def test_stream_response_rejects_invalid_utf8_prompt(self, adapter) -> None:
        with patch(
            "src.shared.python.ai.adapters.bitnet_adapter.subprocess.Popen"
        ) as mock_popen:
            chunks = list(
                adapter.stream_response("bad\udcff", ConversationContext(), [])
            )

        mock_popen.assert_not_called()
        assert len(chunks) == 1
        assert chunks[0].is_final is True
        assert "valid UTF-8 text" in chunks[0].content

    def test_stream_response_rejects_oversize_prompt(self, adapter) -> None:
        message = "x" * (adapter._MAX_PROMPT_BYTES + 1)

        with patch(
            "src.shared.python.ai.adapters.bitnet_adapter.subprocess.Popen"
        ) as mock_popen:
            chunks = list(adapter.stream_response(message, ConversationContext(), []))

        mock_popen.assert_not_called()
        assert len(chunks) == 1
        assert chunks[0].is_final is True
        assert "maximum size" in chunks[0].content
