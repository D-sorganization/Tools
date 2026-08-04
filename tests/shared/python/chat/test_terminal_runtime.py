"""Tests for project-scoped terminal-agent runtime orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
from chat.terminal_runtime import (
    ProcessLaunchRequest,
    TerminalProcessAdapter,
    TerminalRuntimeError,
    TerminalSessionRuntime,
    _build_default_session_env,
)

from chat import (
    TerminalAgentEvent,
    TerminalAgentProviderInfo,
    TerminalAgentSessionRequest,
    TerminalProviderRegistry,
    TerminalRegistryError,
    TerminalShellInfo,
)


def test_terminal_runtime_symbols_are_public() -> None:
    """Consumers can import runtime boundaries from the chat facade."""
    import chat

    expected = {
        "ProcessLaunchRequest",
        "TerminalProcessAdapter",
        "TerminalRuntimeError",
        "TerminalSessionRuntime",
    }

    assert expected.issubset(set(chat.__all__))
    for name in expected:
        assert getattr(chat, name) is not None


class FakeProcessAdapter(TerminalProcessAdapter):
    """In-memory process adapter for runtime contract tests."""

    def __init__(self) -> None:
        self.launches: list[ProcessLaunchRequest] = []
        self.writes: list[tuple[str, str]] = []
        self.resizes: list[tuple[str, int, int]] = []
        self.stopped: list[str] = []
        self._events: dict[str, list[TerminalAgentEvent]] = {}

    def start(self, request: ProcessLaunchRequest) -> str:
        self.launches.append(request)
        process_id = f"proc-{len(self.launches)}"
        self._events[process_id] = [
            TerminalAgentEvent(
                session_id=request.session_id,
                event_type="status",
                data="started",
            )
        ]
        return process_id

    def write(self, process_id: str, text: str) -> None:
        self.writes.append((process_id, text))

    def resize(self, process_id: str, columns: int, rows: int) -> None:
        self.resizes.append((process_id, columns, rows))

    def stop(self, process_id: str) -> None:
        self.stopped.append(process_id)

    def drain_events(self, process_id: str) -> list[TerminalAgentEvent]:
        return self._events.pop(process_id, [])


def _registry() -> TerminalProviderRegistry:
    registry = TerminalProviderRegistry()
    registry.register_shell(
        TerminalShellInfo(
            id="powershell",
            display_name="PowerShell",
            executable="pwsh",
            default_args=["-NoLogo"],
        )
    )
    registry.register_shell(
        TerminalShellInfo(id="bash", display_name="Bash", executable="bash")
    )
    registry.register_provider(
        TerminalAgentProviderInfo(
            id="codex",
            display_name="Codex",
            executable="codex",
            supported_shells=["powershell"],
            launch_args=["--full-auto"],
        )
    )
    return registry


def _request(project_root: Path) -> TerminalAgentSessionRequest:
    return TerminalAgentSessionRequest(
        app_context="tools",
        project_root=project_root,
        shell_id="powershell",
        provider_id="codex",
    )


def test_start_launches_provider_in_resolved_project_root(tmp_path: Path) -> None:
    """Runtime launches provider commands with cwd/env project context."""
    adapter = FakeProcessAdapter()
    runtime = TerminalSessionRuntime(_registry(), adapter)

    session = runtime.start(_request(tmp_path))

    assert session.state == "running"
    assert session.resolved_cwd == tmp_path.resolve()
    launch = adapter.launches[0]
    assert launch.cwd == tmp_path.resolve()
    assert launch.command == ["pwsh", "-NoLogo", "-Command", "codex", "--full-auto"]
    assert launch.env["TOOLS_CHAT_APP_CONTEXT"] == "tools"
    assert launch.env["TOOLS_CHAT_PROJECT_ROOT"] == str(tmp_path.resolve())
    assert launch.env["TOOLS_CHAT_SESSION_ID"] == session.session_id


def test_start_rejects_project_root_outside_allowed_roots(tmp_path: Path) -> None:
    """Allowlist enforcement happens before process launch."""
    allowed_root = tmp_path / "allowed"
    rejected_root = tmp_path / "rejected"
    allowed_root.mkdir()
    rejected_root.mkdir()
    adapter = FakeProcessAdapter()
    runtime = TerminalSessionRuntime(
        _registry(),
        adapter,
        allowed_roots=[allowed_root],
    )

    with pytest.raises(TerminalRuntimeError, match="not under an allowed root"):
        runtime.start(_request(rejected_root))

    assert adapter.launches == []


def test_start_rejects_unsupported_shell_provider_pair(tmp_path: Path) -> None:
    """Runtime delegates shell/provider compatibility to the registry."""
    request = TerminalAgentSessionRequest(
        app_context="tools",
        project_root=tmp_path,
        shell_id="bash",
        provider_id="codex",
    )
    runtime = TerminalSessionRuntime(_registry(), FakeProcessAdapter())

    with pytest.raises(TerminalRegistryError, match="does not support shell"):
        runtime.start(request)


def test_write_resize_stop_use_owning_process(tmp_path: Path) -> None:
    """Lifecycle operations address only the process owned by the session."""
    adapter = FakeProcessAdapter()
    runtime = TerminalSessionRuntime(_registry(), adapter)
    session = runtime.start(_request(tmp_path))

    runtime.write(session.session_id, "hello\n")
    runtime.resize(session.session_id, columns=120, rows=30)
    runtime.stop(session.session_id)

    assert adapter.writes == [("proc-1", "hello\n")]
    assert adapter.resizes == [("proc-1", 120, 30)]
    assert adapter.stopped == ["proc-1"]
    assert runtime.get_session(session.session_id).state == "stopped"


def test_drain_events_returns_normalized_session_events(tmp_path: Path) -> None:
    """Runtime drains normalized events from the process adapter."""
    adapter = FakeProcessAdapter()
    runtime = TerminalSessionRuntime(_registry(), adapter)
    session = runtime.start(_request(tmp_path))

    events = runtime.drain_events(session.session_id)

    assert [(event.event_type, event.data) for event in events] == [
        ("status", "started")
    ]


def test_unknown_session_operations_fail_fast() -> None:
    """Runtime reports invalid session ids before touching the adapter."""
    runtime = TerminalSessionRuntime(_registry(), FakeProcessAdapter())

    with pytest.raises(TerminalRuntimeError, match="unknown terminal session"):
        runtime.write("missing", "text")


def test_secret_like_environment_values_are_not_overridden(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime only adds Tools context vars and preserves caller env mapping."""
    adapter = FakeProcessAdapter()
    base_env: Mapping[str, str] = {
        "CALLER_ENV_VAR": "redacted-test-value",
        "PATH": "test-path",
    }
    runtime = TerminalSessionRuntime(_registry(), adapter, base_env=base_env)
    monkeypatch.delenv("TOOLS_CHAT_SESSION_ID", raising=False)

    runtime.start(_request(tmp_path))

    launch_env = adapter.launches[0].env
    assert launch_env["CALLER_ENV_VAR"] == "redacted-test-value"
    assert launch_env["PATH"] == "test-path"
    assert launch_env["TOOLS_CHAT_SESSION_ID"].startswith("terminal_")


# ---------------------------------------------------------------------------
# Security: credential scrubbing in default session env (issue #2758)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "var_name",
    [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GITHUB_TOKEN",
        "MY_PASSWORD",
        "AWS_SECRET_ACCESS_KEY",
        "STRIPE_PRIVATE_KEY",
    ],
)
def test_default_session_env_excludes_credential_variables(
    monkeypatch: pytest.MonkeyPatch,
    var_name: str,
) -> None:
    """_build_default_session_env never includes credential-like variables.

    Parent-process API keys, tokens, secrets, and passwords must not leak
    into spawned agent subprocesses when no explicit base_env is provided.
    """
    monkeypatch.setenv(var_name, "secret123")

    env = _build_default_session_env()

    assert var_name not in env, (
        f"{var_name!r} must not appear in the default session env"
    )


def test_default_session_env_includes_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """PATH is on the allowlist and must be present in the default session env."""
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    env = _build_default_session_env()

    assert "PATH" in env
    assert env["PATH"] == "/usr/bin:/bin"


def test_runtime_without_base_env_excludes_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TerminalSessionRuntime launched without base_env scrubs credentials.

    The spawned process environment must not contain ANTHROPIC_API_KEY or
    any other credential-like variable inherited from the parent process.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-parent-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-another-secret")

    adapter = FakeProcessAdapter()
    runtime = TerminalSessionRuntime(_registry(), adapter)
    runtime.start(_request(tmp_path))

    launch_env = adapter.launches[0].env
    assert "ANTHROPIC_API_KEY" not in launch_env
    assert "OPENAI_API_KEY" not in launch_env


def test_runtime_with_explicit_base_env_passes_credentials_through(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit base_env is passed through unchanged — caller opts in deliberately.

    When the caller explicitly constructs base_env containing an API key, the
    runtime trusts that intent and forwards the key to the spawned process.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-parent-secret")

    explicit_env: dict[str, str] = {"ANTHROPIC_API_KEY": "sk-caller-provided"}
    adapter = FakeProcessAdapter()
    runtime = TerminalSessionRuntime(_registry(), adapter, base_env=explicit_env)
    runtime.start(_request(tmp_path))

    launch_env = adapter.launches[0].env
    # The caller-provided value must be forwarded; parent env must not override.
    assert launch_env["ANTHROPIC_API_KEY"] == "sk-caller-provided"
