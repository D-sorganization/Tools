"""Unit + live tests for GeminiCliAdapter.

Two layers of coverage — same pattern as ``test_claude_code_adapter.py``:

1. **Unit tests** — subprocess mocked; run everywhere.
2. **Live integration tests** — gated on real ``gemini`` binary being present.

Gemini-specific contracts pinned here:

- ``--skip-trust`` is always passed (the chat UI cannot guarantee a trusted
  workspace).
- Prompt is passed as the ``-p`` argument value (positional), not via stdin
  (the convention differs between Codex / Gemini / Claude Code).
- 120s default timeout.
"""

from __future__ import annotations

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
        _stub = types.ModuleType(_mod_name)
        if _rel_path is not None:
            _stub.__path__ = [str(ROOT / _rel_path)]
        sys.modules[_mod_name] = _stub


_logging_config_stub = sys.modules.setdefault(
    "src.shared.python.logging_pkg.logging_config",
    types.ModuleType("src.shared.python.logging_pkg.logging_config"),
)
_logging_config_stub.get_logger = MagicMock()  # type: ignore[attr-defined]

_env_stub = sys.modules.get("src.shared.python.config.environment")
if not isinstance(_env_stub, types.ModuleType):
    _env_stub = types.ModuleType("src.shared.python.config.environment")
    sys.modules["src.shared.python.config.environment"] = _env_stub
_env_stub.get_env = lambda key, default=None, required=False: default  # type: ignore[attr-defined]
_env_stub.get_env_float = lambda key, default=0.0: float(default)  # type: ignore[attr-defined]


from src.shared.python.ai.adapters.gemini_cli_adapter import (  # noqa: E402
    DEFAULT_GEMINI_CLI_TIMEOUT,
    GeminiCliAdapter,
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
        fake = tmp_path / "gemini.cmd"
        fake.write_text("")
        assert _resolve_binary(str(fake)) == str(fake)

    def test_path_lookup_when_no_explicit(self) -> None:
        with patch("shutil.which", return_value="/usr/local/bin/gemini"):
            assert _resolve_binary() == "/usr/local/bin/gemini"

    def test_returns_none_when_unresolvable(self) -> None:
        with (
            patch("shutil.which", return_value=None),
            patch("pathlib.Path.exists", return_value=False),
        ):
            assert _resolve_binary() is None


class TestGeminiCliAdapter:
    def test_construction_never_fails_when_binary_missing(self) -> None:
        with patch(
            "src.shared.python.ai.adapters.gemini_cli_adapter._resolve_binary",
            return_value=None,
        ):
            adapter = GeminiCliAdapter()
            assert adapter.binary is None
            assert adapter.capabilities.provider_name == "gemini_cli"

    def test_list_models_always_non_empty(self) -> None:
        with patch(
            "src.shared.python.ai.adapters.gemini_cli_adapter._resolve_binary",
            return_value=None,
        ):
            assert GeminiCliAdapter().list_models()

    def test_default_timeout(self) -> None:
        with patch(
            "src.shared.python.ai.adapters.gemini_cli_adapter._resolve_binary",
            return_value="/bin/gemini",
        ):
            assert GeminiCliAdapter().timeout == DEFAULT_GEMINI_CLI_TIMEOUT

    def test_validate_connection_returns_false_when_binary_missing(self) -> None:
        with patch(
            "src.shared.python.ai.adapters.gemini_cli_adapter._resolve_binary",
            return_value=None,
        ):
            ok, msg = GeminiCliAdapter().validate_connection()
            assert ok is False
            assert "not found" in msg.lower()

    def test_validate_connection_runs_version_subprocess(self) -> None:
        fake = MagicMock(returncode=0, stdout="0.39.1\n", stderr="")
        with (
            patch(
                "src.shared.python.ai.adapters.gemini_cli_adapter._resolve_binary",
                return_value="/bin/gemini",
            ),
            patch("subprocess.run", return_value=fake) as mock_run,
        ):
            ok, msg = GeminiCliAdapter().validate_connection()
            assert ok is True
            assert "0.39" in msg
            args, _ = mock_run.call_args
            assert args[0] == ["/bin/gemini", "--version"]

    def test_send_message_passes_skip_trust_flag(self) -> None:
        """Always include ``--skip-trust``.

        Without it the CLI refuses to run from non-whitelisted workspaces,
        which the chat UI cannot guarantee.
        """
        fake = MagicMock(returncode=0, stdout="PONG", stderr="")
        with (
            patch(
                "src.shared.python.ai.adapters.gemini_cli_adapter._resolve_binary",
                return_value="/bin/gemini",
            ),
            patch("subprocess.run", return_value=fake) as mock_run,
        ):
            adapter = GeminiCliAdapter(model="gemini-2.5-flash")
            response = adapter.send_message("ping", ConversationContext(), tools=[])
            assert response.content == "PONG"
            args, _ = mock_run.call_args
            argv = args[0]
            assert "--skip-trust" in argv
            assert "-p" in argv
            assert "--model" in argv
            assert "gemini-2.5-flash" in argv
            # Prompt is the value of -p, the LAST arg.
            assert argv[-1].endswith("User: ping")

    def test_send_message_renders_history(self) -> None:
        fake = MagicMock(returncode=0, stdout="ok", stderr="")
        ctx = ConversationContext()
        ctx.messages = [  # type: ignore[attr-defined]
            MagicMock(role="user", content="first turn"),
            MagicMock(role="assistant", content="first reply"),
        ]
        with (
            patch(
                "src.shared.python.ai.adapters.gemini_cli_adapter._resolve_binary",
                return_value="/bin/gemini",
            ),
            patch("subprocess.run", return_value=fake) as mock_run,
        ):
            GeminiCliAdapter().send_message("second turn", ctx, tools=[])
            args, _ = mock_run.call_args
            prompt = args[0][-1]
            assert "User: first turn" in prompt
            assert "Assistant: first reply" in prompt
            assert prompt.endswith("User: second turn")

    def test_send_message_raises_when_binary_missing(self) -> None:
        with patch(
            "src.shared.python.ai.adapters.gemini_cli_adapter._resolve_binary",
            return_value=None,
        ):
            with pytest.raises(AIConnectionError, match="not found"):
                GeminiCliAdapter().send_message("hi", ConversationContext(), tools=[])

    def test_send_message_maps_timeout(self) -> None:
        with (
            patch(
                "src.shared.python.ai.adapters.gemini_cli_adapter._resolve_binary",
                return_value="/bin/gemini",
            ),
            patch(
                "subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="gemini", timeout=1.0),
            ),
        ):
            with pytest.raises(AITimeoutError):
                GeminiCliAdapter(timeout=1.0).send_message(
                    "hi", ConversationContext(), tools=[]
                )

    def test_send_message_maps_nonzero_exit_to_provider_error(self) -> None:
        fake = MagicMock(returncode=2, stdout="", stderr="not authenticated")
        with (
            patch(
                "src.shared.python.ai.adapters.gemini_cli_adapter._resolve_binary",
                return_value="/bin/gemini",
            ),
            patch("subprocess.run", return_value=fake),
        ):
            with pytest.raises(AIProviderError, match="not authenticated"):
                GeminiCliAdapter().send_message("hi", ConversationContext(), tools=[])

    def test_send_message_rejects_empty_message(self) -> None:
        with patch(
            "src.shared.python.ai.adapters.gemini_cli_adapter._resolve_binary",
            return_value="/bin/gemini",
        ):
            with pytest.raises(ValueError):
                GeminiCliAdapter().send_message("", ConversationContext(), tools=[])


# ─── Live integration tests (skip when CLI missing) ────────────────────


_HAS_GEMINI = _resolve_binary() is not None


@pytest.mark.skipif(
    not _HAS_GEMINI,
    reason="Gemini CLI not installed; skipping live integration test.",
)
class TestLiveGeminiCli:
    def test_version_probe_succeeds(self) -> None:
        adapter = GeminiCliAdapter(timeout=15.0)
        ok, msg = adapter.validate_connection()
        assert ok, f"Live validate_connection failed: {msg}"

    @pytest.mark.slow
    def test_real_chat_round_trip(self) -> None:
        """Real Gemini CLI round trip."""
        adapter = GeminiCliAdapter(timeout=60.0)
        response = adapter.send_message(
            "Reply with the single word PONG and nothing else.",
            ConversationContext(),
            tools=[],
        )
        assert "PONG" in response.content.upper()
        assert response.metadata.get("provider") == "gemini_cli"
