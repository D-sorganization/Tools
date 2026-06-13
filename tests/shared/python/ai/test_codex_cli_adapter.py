# ruff: noqa: E501
"""Unit + live tests for CodexCliAdapter.

Two layers of coverage — same pattern as ``test_claude_code_adapter.py``:

1. **Unit tests** — subprocess mocked; run everywhere.
2. **Live integration tests** — gated on real ``codex`` binary being present.

Codex-specific contracts pinned here:

- ``codex exec --skip-git-repo-check`` is always passed (the chat UI cannot
  guarantee a trusted git directory).
- Telemetry preamble lines like ``[2026-05-17T23:13:03] thinking…`` are
  stripped from the response.
- Cold-start timeout default is 180s.

Bootstrap pattern mirrors ``test_ollama_adapter.py`` to work around the broken
``src.shared`` package import on this Python version.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.ai.adapters.codex_cli_adapter import (  # noqa: E402
    DEFAULT_CODEX_CLI_TIMEOUT,
    CodexCliAdapter,
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
    def test_explicit_path_returned_when_exists(self, tmp_path) -> None:
        fake = tmp_path / "codex.cmd"
        fake.write_text("")
        assert _resolve_binary(str(fake)) == str(fake)

    def test_path_lookup_when_no_explicit(self) -> None:
        with patch("shutil.which", return_value="/usr/local/bin/codex"):
            assert _resolve_binary() == "/usr/local/bin/codex"

    def test_returns_none_when_unresolvable(self) -> None:
        with (
            patch("shutil.which", return_value=None),
            patch("pathlib.Path.exists", return_value=False),
        ):
            assert _resolve_binary() is None


class TestCodexCliAdapter:
    def test_construction_never_fails_when_binary_missing(self) -> None:
        with patch(
            "src.shared.python.ai.adapters.codex_cli_adapter._resolve_binary",
            return_value=None,
        ):
            adapter = CodexCliAdapter()
            assert adapter.binary is None
            assert adapter.capabilities.provider_name == "codex_cli"

    def test_list_models_always_non_empty(self) -> None:
        with patch(
            "src.shared.python.ai.adapters.codex_cli_adapter._resolve_binary",
            return_value=None,
        ):
            assert CodexCliAdapter().list_models()

    def test_default_timeout(self) -> None:
        with patch(
            "src.shared.python.ai.adapters.codex_cli_adapter._resolve_binary",
            return_value="/bin/codex",
        ):
            assert CodexCliAdapter().timeout == DEFAULT_CODEX_CLI_TIMEOUT

    def test_validate_connection_returns_false_when_binary_missing(self) -> None:
        with patch(
            "src.shared.python.ai.adapters.codex_cli_adapter._resolve_binary",
            return_value=None,
        ):
            ok, msg = CodexCliAdapter().validate_connection()
            assert ok is False
            assert "not found" in msg.lower()

    def test_validate_connection_runs_version_subprocess(self) -> None:
        fake = MagicMock(returncode=0, stdout="codex-cli 0.118.0\n", stderr="")
        with (
            patch(
                "src.shared.python.ai.adapters.codex_cli_adapter._resolve_binary",
                return_value="/bin/codex",
            ),
            patch("subprocess.run", return_value=fake) as mock_run,
        ):
            ok, msg = CodexCliAdapter().validate_connection()
            assert ok is True
            assert "0.118" in msg
            args, _ = mock_run.call_args
            assert args[0] == ["/bin/codex", "--version"]

    def test_send_message_passes_skip_git_repo_flag(self) -> None:
        """Always include ``--skip-git-repo-check``.

        Without it Codex refuses to run from non-git directories, which
        the chat UI cannot guarantee. Regressing this breaks chat for
        every user not running from inside a git repo.
        """
        fake = MagicMock(returncode=0, stdout="PONG", stderr="")
        with (
            patch(
                "src.shared.python.ai.adapters.codex_cli_adapter._resolve_binary",
                return_value="/bin/codex",
            ),
            patch("subprocess.run", return_value=fake) as mock_run,
        ):
            adapter = CodexCliAdapter(model="gpt-5-mini")
            response = adapter.send_message("ping", ConversationContext(), tools=[])
            assert response.content == "PONG"
            args, _ = mock_run.call_args
            argv = args[0]
            assert "exec" in argv
            assert "--skip-git-repo-check" in argv
            assert "--model" in argv
            assert "gpt-5-mini" in argv
            # Prompt is positional, last arg.
            assert argv[-1].endswith("User: ping")

    def test_send_message_strips_telemetry_preamble(self) -> None:
        """Codex prefixes responses with timestamp banners. Drop them."""
        noisy = (
            "[2026-05-17T23:13:03] thinking…\n"
            "[2026-05-17T23:13:05] generating…\n"
            "\n"
            "Hello! Your real answer is here.\n"
        )
        fake = MagicMock(returncode=0, stdout=noisy, stderr="")
        with (
            patch(
                "src.shared.python.ai.adapters.codex_cli_adapter._resolve_binary",
                return_value="/bin/codex",
            ),
            patch("subprocess.run", return_value=fake),
        ):
            response = CodexCliAdapter().send_message(
                "hi", ConversationContext(), tools=[]
            )
            assert response.content == "Hello! Your real answer is here."
            assert "thinking" not in response.content
            assert "[2026" not in response.content

    def test_send_message_renders_history(self) -> None:
        fake = MagicMock(returncode=0, stdout="ok", stderr="")
        ctx = ConversationContext()
        ctx.messages = [  # type: ignore[attr-defined]
            MagicMock(role="user", content="first turn"),
            MagicMock(role="assistant", content="first reply"),
        ]
        with (
            patch(
                "src.shared.python.ai.adapters.codex_cli_adapter._resolve_binary",
                return_value="/bin/codex",
            ),
            patch("subprocess.run", return_value=fake) as mock_run,
        ):
            CodexCliAdapter().send_message("second turn", ctx, tools=[])
            args, _ = mock_run.call_args
            prompt = args[0][-1]
            assert "User: first turn" in prompt
            assert "Assistant: first reply" in prompt
            assert prompt.endswith("User: second turn")

    def test_send_message_raises_when_binary_missing(self) -> None:
        with patch(
            "src.shared.python.ai.adapters.codex_cli_adapter._resolve_binary",
            return_value=None,
        ):
            with pytest.raises(AIConnectionError, match="not found"):
                CodexCliAdapter().send_message("hi", ConversationContext(), tools=[])

    def test_send_message_maps_timeout(self) -> None:
        with (
            patch(
                "src.shared.python.ai.adapters.codex_cli_adapter._resolve_binary",
                return_value="/bin/codex",
            ),
            patch(
                "subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="codex", timeout=1.0),
            ),
        ):
            with pytest.raises(AITimeoutError):
                CodexCliAdapter(timeout=1.0).send_message(
                    "hi", ConversationContext(), tools=[]
                )

    def test_send_message_maps_nonzero_exit_to_provider_error(self) -> None:
        fake = MagicMock(returncode=2, stdout="", stderr="not authenticated")
        with (
            patch(
                "src.shared.python.ai.adapters.codex_cli_adapter._resolve_binary",
                return_value="/bin/codex",
            ),
            patch("subprocess.run", return_value=fake),
        ):
            with pytest.raises(AIProviderError, match="not authenticated"):
                CodexCliAdapter().send_message("hi", ConversationContext(), tools=[])

    def test_send_message_rejects_empty_message(self) -> None:
        with patch(
            "src.shared.python.ai.adapters.codex_cli_adapter._resolve_binary",
            return_value="/bin/codex",
        ):
            with pytest.raises(ValueError):
                CodexCliAdapter().send_message("", ConversationContext(), tools=[])


# ─── Live integration tests (skip when CLI missing) ────────────────────


_HAS_CODEX = _resolve_binary() is not None


@pytest.mark.skipif(
    not _HAS_CODEX,
    reason="Codex CLI not installed; skipping live integration test.",
)
class TestLiveCodexCli:
    def test_version_probe_succeeds(self) -> None:
        adapter = CodexCliAdapter(timeout=15.0)
        ok, msg = adapter.validate_connection()
        assert ok, f"Live validate_connection failed: {msg}"

    @pytest.mark.slow
    def test_real_chat_round_trip(self) -> None:
        """Real Codex round trip — slow (cold-start can be 30-60s)."""
        adapter = CodexCliAdapter(timeout=120.0)
        response = adapter.send_message(
            "Reply with the single word PONG and nothing else.",
            ConversationContext(),
            tools=[],
        )
        assert "PONG" in response.content.upper()
        assert response.metadata.get("provider") == "codex_cli"
