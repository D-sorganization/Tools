"""Tests for project-scoped terminal-agent runtime orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
from chat import (
    TerminalAgentEvent,
    TerminalAgentProviderInfo,
    TerminalAgentSessionRequest,
    TerminalProviderRegistry,
    TerminalRegistryError,
    TerminalShellInfo,
)
from chat.terminal_runtime import (
    ProcessLaunchRequest,
    TerminalProcessAdapter,
    TerminalRuntimeError,
    TerminalSessionRuntime,
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


def test_stop_does_not_mutate_held_reference_to_old_info(tmp_path: Path) -> None:
    """Stopping a session must not mutate a previously captured info reference.

    TerminalAgentSessionInfo is frozen; stop() must produce a new instance via
    model_copy so callers that stored the old reference see its original state.
    """
    adapter = FakeProcessAdapter()
    runtime = TerminalSessionRuntime(_registry(), adapter)
    runtime.start(_request(tmp_path))
    session_id = runtime.start(_request(tmp_path)).session_id

    old_info = runtime.get_session(session_id)
    assert old_info.state == "running"

    runtime.stop(session_id)

    # The held reference must be unchanged (frozen model — no mutation)
    assert old_info.state != "stopped"
    # The registry must reflect the new state
    assert runtime.get_session(session_id).state == "stopped"


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
