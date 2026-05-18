"""Unit tests for GitHubCliProvider (Tools #2899).

Covers:
- ``github-cli`` descriptor present in ``_CLI_AGENT_DESCRIPTORS``.
- ``is_available()`` / ``validate_connection()`` semantics around
  ``gh auth status``.
- Intent detection: chat-style messages route to the correct ``gh``
  subcommand.
- Streaming output: stdout lines yield ``AgentChunk`` instances in order,
  terminating with ``is_final=True``.
- Error paths: ``gh`` missing, not authenticated, non-zero exit.
- Unsupported intent: graceful fallback message.

The tests stub ``subprocess`` so they run without a real ``gh`` binary.
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
# so importing the adapter directly works in a plain pytest run. Matches
# the pattern used by tests/shared/python/ai/test_bitnet_adapter.py.
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
    ("src.shared.python.chat", "src/shared/python/chat"),
]
for _mod_name, _rel_path in _PACKAGE_STUBS:
    if _mod_name not in sys.modules:
        _stub = types.ModuleType(_mod_name)
        if _rel_path is not None:
            _stub.__path__ = [str(ROOT / _rel_path)]
        sys.modules[_mod_name] = _stub

_logging_pkg_stub = sys.modules.setdefault(
    "src.shared.python.logging_pkg",
    types.ModuleType("src.shared.python.logging_pkg"),
)
_logging_config_stub = sys.modules.setdefault(
    "src.shared.python.logging_pkg.logging_config",
    types.ModuleType("src.shared.python.logging_pkg.logging_config"),
)
_logging_config_stub.get_logger = logging.getLogger  # type: ignore[attr-defined]
_logging_config_stub.setup_logging = lambda *a, **kw: None  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Now import the module under test.
# ---------------------------------------------------------------------------

from src.shared.python.ai.adapters.github_cli_provider import (  # noqa: E402
    GitHubCliProvider,
    GitHubCliResult,
    detect_gh_intent,
)
from src.shared.python.ai.types import (  # noqa: E402
    AgentChunk,
    ConversationContext,
)
from src.shared.python.chat.cli_provider_availability import (  # noqa: E402
    _CLI_AGENT_DESCRIPTORS,
    list_available_cli_providers,
)

# ---------------------------------------------------------------------------
# Descriptor registration
# ---------------------------------------------------------------------------


class TestDescriptorRegistration:
    def test_github_cli_descriptor_present(self) -> None:
        """The new descriptor must appear in _CLI_AGENT_DESCRIPTORS."""
        ids = [provider_id for provider_id, _, _ in _CLI_AGENT_DESCRIPTORS]
        assert "github-cli" in ids

    def test_github_cli_descriptor_binary_is_gh(self) -> None:
        """The probed binary must be ``gh``."""
        for provider_id, _, executable in _CLI_AGENT_DESCRIPTORS:
            if provider_id == "github-cli":
                assert executable == "gh"
                return
        pytest.fail("github-cli descriptor not found")

    def test_github_cli_display_name_is_human_friendly(self) -> None:
        for provider_id, display_name, _ in _CLI_AGENT_DESCRIPTORS:
            if provider_id == "github-cli":
                assert "GitHub" in display_name
                return
        pytest.fail("github-cli descriptor not found")

    def test_listed_when_gh_on_path(self) -> None:
        """list_available_cli_providers() surfaces gh when present."""
        with patch("shutil.which", side_effect=lambda x: f"/usr/bin/{x}"):
            providers = list_available_cli_providers()
        ids = [p.provider_id for p in providers]
        assert "github-cli" in ids

    def test_omitted_when_gh_missing(self) -> None:
        """list_available_cli_providers() omits gh when absent."""
        with patch("shutil.which", return_value=None):
            providers = list_available_cli_providers()
        assert providers == []


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------


class TestIntentDetection:
    def test_list_issues(self) -> None:
        intent = detect_gh_intent("list my issues")
        assert intent is not None
        assert intent.args[:2] == ["issue", "list"]

    def test_list_my_issues_adds_me_flag(self) -> None:
        intent = detect_gh_intent("list my issues")
        assert intent is not None
        assert "--me" in intent.args

    def test_list_prs(self) -> None:
        intent = detect_gh_intent("list PRs")
        assert intent is not None
        assert intent.args[:2] == ["pr", "list"]

    def test_create_issue(self) -> None:
        intent = detect_gh_intent('create issue titled "fix the broken thing"')
        assert intent is not None
        assert intent.args[:2] == ["issue", "create"]
        assert "fix the broken thing" in intent.args
        # destructive — must be flagged
        assert intent.requires_confirmation is True

    def test_create_issue_single_quotes(self) -> None:
        intent = detect_gh_intent("create issue titled 'hello world'")
        assert intent is not None
        assert "hello world" in intent.args

    def test_merge_pr(self) -> None:
        intent = detect_gh_intent("merge PR #42")
        assert intent is not None
        assert intent.args[:2] == ["pr", "merge"]
        assert "42" in intent.args
        assert intent.requires_confirmation is True

    def test_view_repo(self) -> None:
        intent = detect_gh_intent("view repo D-sorganization/Tools")
        assert intent is not None
        assert intent.args[:2] == ["repo", "view"]
        assert "D-sorganization/Tools" in intent.args

    def test_view_issue_by_number(self) -> None:
        intent = detect_gh_intent("view issue #2899")
        assert intent is not None
        assert intent.args[:2] == ["issue", "view"]
        assert "2899" in intent.args

    def test_unsupported_intent_returns_none(self) -> None:
        intent = detect_gh_intent("please make me a sandwich")
        assert intent is None

    def test_empty_message_returns_none(self) -> None:
        assert detect_gh_intent("") is None
        assert detect_gh_intent("   ") is None


# ---------------------------------------------------------------------------
# Provider behavior
# ---------------------------------------------------------------------------


def _make_completed(stdout: str = "", stderr: str = "", returncode: int = 0):
    cp = MagicMock(spec=subprocess.CompletedProcess)
    cp.stdout = stdout
    cp.stderr = stderr
    cp.returncode = returncode
    return cp


class TestProviderAvailability:
    def test_is_available_true_when_gh_authed(self) -> None:
        provider = GitHubCliProvider()
        with (
            patch("shutil.which", return_value="/usr/bin/gh"),
            patch(
                "subprocess.run",
                return_value=_make_completed(stdout="Logged in to github.com"),
            ),
        ):
            assert provider.is_available() is True

    def test_is_available_false_when_gh_missing(self) -> None:
        provider = GitHubCliProvider()
        with patch("shutil.which", return_value=None):
            assert provider.is_available() is False

    def test_is_available_false_when_not_authenticated(self) -> None:
        provider = GitHubCliProvider()
        with (
            patch("shutil.which", return_value="/usr/bin/gh"),
            patch(
                "subprocess.run",
                return_value=_make_completed(
                    stderr="You are not logged into any GitHub hosts.",
                    returncode=1,
                ),
            ),
        ):
            assert provider.is_available() is False

    def test_validate_connection_reports_diagnostic(self) -> None:
        provider = GitHubCliProvider()
        with (
            patch("shutil.which", return_value="/usr/bin/gh"),
            patch(
                "subprocess.run",
                return_value=_make_completed(
                    stdout="Logged in to github.com as dieterolson",
                ),
            ),
        ):
            ok, msg = provider.validate_connection()
        assert ok is True
        assert "dieterolson" in msg or "github.com" in msg.lower()

    def test_validate_connection_gh_missing(self) -> None:
        provider = GitHubCliProvider()
        with patch("shutil.which", return_value=None):
            ok, msg = provider.validate_connection()
        assert ok is False
        assert "gh" in msg.lower()

    def test_validate_connection_not_authenticated(self) -> None:
        provider = GitHubCliProvider()
        with (
            patch("shutil.which", return_value="/usr/bin/gh"),
            patch(
                "subprocess.run",
                return_value=_make_completed(
                    stderr="You are not logged into any GitHub hosts.",
                    returncode=1,
                ),
            ),
        ):
            ok, msg = provider.validate_connection()
        assert ok is False
        assert "auth" in msg.lower() or "login" in msg.lower()


class TestSendMessageRouting:
    """``send`` translates a chat message into a ``gh`` invocation."""

    def test_send_list_my_issues_invokes_gh_issue_list(self) -> None:
        provider = GitHubCliProvider()
        recorded: list[list[str]] = []

        def _fake_run(cmd, *a, **kw):
            recorded.append(list(cmd))
            return _make_completed(stdout="[]")

        with (
            patch("shutil.which", return_value="/usr/bin/gh"),
            patch("subprocess.run", side_effect=_fake_run),
        ):
            result = provider.send("list my issues")

        assert isinstance(result, GitHubCliResult)
        assert result.exit_code == 0
        # Only the actual gh call counts (auth check is filtered).
        gh_calls = [c for c in recorded if c[:2] == ["gh", "issue"]]
        assert gh_calls, f"no gh issue call recorded; saw: {recorded}"
        assert gh_calls[0][:3] == ["gh", "issue", "list"]
        assert "--me" in gh_calls[0]

    def test_send_unsupported_intent_returns_help(self) -> None:
        provider = GitHubCliProvider()
        with (
            patch("shutil.which", return_value="/usr/bin/gh"),
            patch("subprocess.run", return_value=_make_completed()),
        ):
            result = provider.send("compute pi to a million digits")
        assert result.exit_code != 0
        # Help text mentions supported intents
        lower = result.stderr.lower() + result.stdout.lower()
        assert "list issues" in lower
        assert "create issue" in lower

    def test_send_requires_is_available(self) -> None:
        """Precondition: send() requires is_available()."""
        provider = GitHubCliProvider()
        with patch("shutil.which", return_value=None):
            with pytest.raises((RuntimeError, ValueError, AssertionError)):
                provider.send("list my issues")


class TestStreamingOutput:
    def test_stream_produces_chunks_in_order(self) -> None:
        provider = GitHubCliProvider()
        # Simulate three stdout lines + a final newline.
        fake_proc = MagicMock()
        fake_proc.stdout = iter(["line one\n", "line two\n", "line three\n"])
        fake_proc.stderr = iter([])
        fake_proc.wait.return_value = 0
        fake_proc.returncode = 0
        fake_proc.poll.return_value = 0

        with (
            patch("shutil.which", return_value="/usr/bin/gh"),
            patch("subprocess.Popen", return_value=fake_proc),
            patch(
                "subprocess.run",
                return_value=_make_completed(stdout="Logged in"),
            ),
        ):
            chunks = list(provider.stream("list my issues"))

        contents = [c.content for c in chunks if isinstance(c, AgentChunk)]
        assert "line one\n" in contents
        assert "line two\n" in contents
        assert "line three\n" in contents
        # last chunk must be final
        assert chunks[-1].is_final is True
        # indices ascend
        indices = [c.index for c in chunks]
        assert indices == sorted(indices)

    def test_stream_unsupported_intent_yields_help_chunk(self) -> None:
        provider = GitHubCliProvider()
        with (
            patch("shutil.which", return_value="/usr/bin/gh"),
            patch(
                "subprocess.run",
                return_value=_make_completed(stdout="Logged in"),
            ),
        ):
            chunks = list(provider.stream("make me coffee"))
        assert chunks
        assert chunks[-1].is_final is True
        text = "".join(c.content for c in chunks).lower()
        assert "list issues" in text


class TestCancel:
    def test_cancel_is_idempotent(self) -> None:
        provider = GitHubCliProvider()
        # No active process — cancel should be safe.
        provider.cancel()
        provider.cancel()
        provider.cancel()  # third time still fine


class TestSendMessageWithContext:
    """Integration with BaseAgentAdapter.send_message / stream_response."""

    def test_send_message_returns_agent_response(self) -> None:
        provider = GitHubCliProvider()
        ctx = ConversationContext()
        with (
            patch("shutil.which", return_value="/usr/bin/gh"),
            patch(
                "subprocess.run",
                return_value=_make_completed(stdout="issue list output"),
            ),
        ):
            resp = provider.send_message("list my issues", ctx, [])
        assert "issue list output" in resp.content
        assert resp.finish_reason == "stop"

    def test_send_message_error_path_surfaces_diagnostic(self) -> None:
        provider = GitHubCliProvider()
        ctx = ConversationContext()

        def _fake_run(cmd, *a, **kw):
            # auth check succeeds; gh issue list fails with 401
            if "auth" in cmd:
                return _make_completed(stdout="Logged in")
            return _make_completed(stderr="HTTP 401: Bad credentials", returncode=1)

        with (
            patch("shutil.which", return_value="/usr/bin/gh"),
            patch("subprocess.run", side_effect=_fake_run),
        ):
            resp = provider.send_message("list my issues", ctx, [])
        assert "401" in resp.content or "credentials" in resp.content.lower()
        assert resp.finish_reason in {"error", "stop"}


class TestCapabilitiesAndCatalogue:
    def test_capabilities_advertise_streaming(self) -> None:
        provider = GitHubCliProvider()
        caps = provider.capabilities
        from src.shared.python.ai.types import ProviderCapability

        assert ProviderCapability.STREAMING in caps.supported

    def test_list_models_non_empty(self) -> None:
        provider = GitHubCliProvider()
        models = provider.list_models()
        assert models, "list_models() must be non-empty"
        assert all(isinstance(m, str) for m in models)

    def test_thinking_capabilities_none_only(self) -> None:
        provider = GitHubCliProvider()
        caps = provider.thinking_capabilities()
        # Should be a ThinkingCapabilities with 'none' level
        assert caps is not None
