# ruff: noqa: E501
"""Unit + live tests for ClaudeCodeAdapter.

Two layers of coverage so the adapter stays wired up:

1. **Unit tests** — subprocess is mocked; run everywhere. These pin the
   adapter's contract (argument shape, prompt rendering, error mapping,
   timeout handling).
2. **Live integration tests** — gated on the real ``claude`` binary being
   present; auto-skip when the CLI is missing.

Bootstrap pattern mirrors ``test_ollama_adapter.py`` to work around the broken
``src.shared`` package import on this Python version.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.ai.adapters.claude_code_adapter import (  # noqa: E402
    DEFAULT_CLAUDE_CODE_TIMEOUT,
    ClaudeCodeAdapter,
    _resolve_binary,
)
from src.shared.python.ai.exceptions import (  # noqa: E402
    AIConnectionError,
    AIProviderError,
    AITimeoutError,
)
from src.shared.python.ai.types import ConversationContext  # noqa: E402

# ─── Unit tests (always run) ──────────────────────────────────────────


class TestResolveBinary:
    """``_resolve_binary`` finds the CLI across plausible install locations."""

    def test_explicit_path_returned_when_exists(self, tmp_path) -> None:
        fake = tmp_path / "claude.exe"
        fake.write_text("")
        assert _resolve_binary(str(fake)) == str(fake)

    def test_explicit_path_falls_through_when_missing(self) -> None:
        with patch("shutil.which", return_value=None):
            with patch("pathlib.Path.exists", return_value=False):
                assert _resolve_binary("C:/nope/claude.exe") is None

    def test_path_lookup_when_no_explicit(self) -> None:
        with patch("shutil.which", return_value="/usr/local/bin/claude"):
            assert _resolve_binary() == "/usr/local/bin/claude"

    def test_returns_none_when_unresolvable(self) -> None:
        with (
            patch("shutil.which", return_value=None),
            patch("pathlib.Path.exists", return_value=False),
        ):
            assert _resolve_binary() is None


class TestClaudeCodeAdapter:
    """Adapter behavior under mocked subprocess."""

    def test_construction_never_fails_when_binary_missing(self) -> None:
        """The chat UI calls ``ClaudeCodeAdapter()`` eagerly to populate the
        provider list; raising here would break the dropdown for users
        who haven't installed Claude Code.
        """
        with patch(
            "src.shared.python.ai.adapters.claude_code_adapter._resolve_binary",
            return_value=None,
        ):
            adapter = ClaudeCodeAdapter()
            assert adapter.binary is None
            assert adapter.capabilities.provider_name == "claude_code"

    def test_list_models_always_non_empty(self) -> None:
        with patch(
            "src.shared.python.ai.adapters.claude_code_adapter._resolve_binary",
            return_value=None,
        ):
            adapter = ClaudeCodeAdapter()
            models = adapter.list_models()
            assert isinstance(models, list)
            assert len(models) > 0
            assert all(isinstance(m, str) and m for m in models)

    def test_default_timeout(self) -> None:
        with patch(
            "src.shared.python.ai.adapters.claude_code_adapter._resolve_binary",
            return_value="/bin/claude",
        ):
            adapter = ClaudeCodeAdapter()
            assert adapter.timeout == DEFAULT_CLAUDE_CODE_TIMEOUT

    def test_validate_connection_returns_false_when_binary_missing(self) -> None:
        with patch(
            "src.shared.python.ai.adapters.claude_code_adapter._resolve_binary",
            return_value=None,
        ):
            adapter = ClaudeCodeAdapter()
            ok, msg = adapter.validate_connection()
            assert ok is False
            assert "not found" in msg.lower()

    def test_validate_connection_runs_version_subprocess(self) -> None:
        fake = MagicMock(returncode=0, stdout="2.1.91 (Claude Code)\n", stderr="")
        with (
            patch(
                "src.shared.python.ai.adapters.claude_code_adapter._resolve_binary",
                return_value="/bin/claude",
            ),
            patch("subprocess.run", return_value=fake) as mock_run,
        ):
            adapter = ClaudeCodeAdapter()
            ok, msg = adapter.validate_connection()
            assert ok is True
            assert "2.1.91" in msg
            args, _ = mock_run.call_args
            assert args[0] == ["/bin/claude", "--version"]

    def test_validate_connection_handles_nonzero_exit(self) -> None:
        fake = MagicMock(returncode=2, stdout="", stderr="auth missing")
        with (
            patch(
                "src.shared.python.ai.adapters.claude_code_adapter._resolve_binary",
                return_value="/bin/claude",
            ),
            patch("subprocess.run", return_value=fake),
        ):
            adapter = ClaudeCodeAdapter()
            ok, msg = adapter.validate_connection()
            assert ok is False
            assert "auth missing" in msg

    def test_send_message_passes_prompt_via_stdin(self) -> None:
        fake = MagicMock(returncode=0, stdout="PONG\n", stderr="")
        with (
            patch(
                "src.shared.python.ai.adapters.claude_code_adapter._resolve_binary",
                return_value="/bin/claude",
            ),
            patch("subprocess.run", return_value=fake) as mock_run,
        ):
            adapter = ClaudeCodeAdapter(model="sonnet")
            response = adapter.send_message("ping", ConversationContext(), tools=[])
            assert response.content == "PONG"
            assert response.metadata.get("provider") == "claude_code"
            args, kwargs = mock_run.call_args
            assert args[0] == ["/bin/claude", "-p", "--model", "sonnet"]
            # Prompt MUST go via stdin (-p without an argument), not as argv.
            assert kwargs["input"] == "User: ping"

    def test_send_message_renders_history(self) -> None:
        fake = MagicMock(returncode=0, stdout="ok", stderr="")
        ctx = ConversationContext()
        # Build history without depending on Message-class internals.
        ctx.messages = [  # type: ignore[attr-defined]
            MagicMock(role="user", content="hello"),
            MagicMock(role="assistant", content="hi there"),
        ]
        with (
            patch(
                "src.shared.python.ai.adapters.claude_code_adapter._resolve_binary",
                return_value="/bin/claude",
            ),
            patch("subprocess.run", return_value=fake) as mock_run,
        ):
            adapter = ClaudeCodeAdapter()
            adapter.send_message("follow-up", ctx, tools=[])
            _, kwargs = mock_run.call_args
            prompt = kwargs["input"]
            assert "User: hello" in prompt
            assert "Assistant: hi there" in prompt
            assert "User: follow-up" in prompt
            # Order matters: history first, current message last.
            assert prompt.rfind("User: follow-up") > prompt.find("User: hello")

    def test_send_message_raises_when_binary_missing(self) -> None:
        with patch(
            "src.shared.python.ai.adapters.claude_code_adapter._resolve_binary",
            return_value=None,
        ):
            adapter = ClaudeCodeAdapter()
            with pytest.raises(AIConnectionError, match="not found"):
                adapter.send_message("hi", ConversationContext(), tools=[])

    def test_send_message_maps_timeout(self) -> None:
        with (
            patch(
                "src.shared.python.ai.adapters.claude_code_adapter._resolve_binary",
                return_value="/bin/claude",
            ),
            patch(
                "subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=1.0),
            ),
        ):
            adapter = ClaudeCodeAdapter(timeout=1.0)
            with pytest.raises(AITimeoutError):
                adapter.send_message("hi", ConversationContext(), tools=[])

    def test_send_message_maps_nonzero_exit_to_provider_error(self) -> None:
        fake = MagicMock(returncode=1, stdout="", stderr="rate limited")
        with (
            patch(
                "src.shared.python.ai.adapters.claude_code_adapter._resolve_binary",
                return_value="/bin/claude",
            ),
            patch("subprocess.run", return_value=fake),
        ):
            adapter = ClaudeCodeAdapter()
            with pytest.raises(AIProviderError, match="rate limited"):
                adapter.send_message("hi", ConversationContext(), tools=[])

    def test_send_message_rejects_empty_message(self) -> None:
        with patch(
            "src.shared.python.ai.adapters.claude_code_adapter._resolve_binary",
            return_value="/bin/claude",
        ):
            adapter = ClaudeCodeAdapter()
            with pytest.raises(ValueError):
                adapter.send_message("   ", ConversationContext(), tools=[])

    def test_stream_response_yields_single_final_chunk(self) -> None:
        fake = MagicMock(returncode=0, stdout="streamed", stderr="")
        with (
            patch(
                "src.shared.python.ai.adapters.claude_code_adapter._resolve_binary",
                return_value="/bin/claude",
            ),
            patch("subprocess.run", return_value=fake),
        ):
            adapter = ClaudeCodeAdapter()
            chunks = list(
                adapter.stream_response("hi", ConversationContext(), tools=[])
            )
            assert len(chunks) == 1
            assert chunks[0].is_final
            assert chunks[0].content == "streamed"


# ─── Live integration tests (skip when CLI missing) ────────────────────


_HAS_CLAUDE = _resolve_binary() is not None


@pytest.mark.skipif(
    not _HAS_CLAUDE,
    reason="Claude Code CLI not installed; skipping live integration test.",
)
class TestLiveClaudeCode:
    """Real CLI tests.

    Auto-skip when ``claude`` is not on PATH so CI doesn't fail on machines
    without the agent installed. On developer machines and dashboard hosts
    that *do* have it, these guard against silent breakage of the live path.
    """

    def test_version_probe_succeeds(self) -> None:
        adapter = ClaudeCodeAdapter(timeout=15.0)
        ok, msg = adapter.validate_connection()
        assert ok, f"Live validate_connection failed: {msg}"

    @pytest.mark.slow
    def test_real_chat_round_trip(self) -> None:
        """Real ``claude -p`` round trip with a known-stable prompt."""
        adapter = ClaudeCodeAdapter(timeout=60.0)
        response = adapter.send_message(
            "Reply with the single word PONG and nothing else.",
            ConversationContext(),
            tools=[],
        )
        assert "PONG" in response.content.upper()
        assert response.metadata.get("provider") == "claude_code"
