"""#6023 C1 / #5727: timeout paths kill the CLI process group with a real PID."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from shared.python.ai.adapters.cli_subprocess_runner import (
    CliSubprocessTimeoutError,
    cli_subprocess_pid_from_error,
    invoke_cli_subprocess_for_adapter,
    terminate_cli_process_group,
    terminate_cli_subprocess_after_error,
)
from shared.python.ai.exceptions import AITimeoutError
from shared.python.ai.types import ConversationContext
from tests.unit.ai.adapters.test_cli_subprocess_runner_oracles import (
    install_fake_cli,
)

pytestmark = pytest.mark.unit


def test_cli_subprocess_pid_from_error_walks_exception_cause_chain() -> None:
    inner = CliSubprocessTimeoutError(["/bin/fake"], 1.0, pid=424242)
    outer = AITimeoutError("outer", provider="claude_code")
    outer.__cause__ = inner
    assert cli_subprocess_pid_from_error(outer) == 424242


def test_terminate_cli_subprocess_after_error_uses_popen_without_pid_on_exc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backup kill must use the live handle when exceptions omit ``pid``."""
    calls: list[int | None] = []
    proc = MagicMock()
    proc.poll.return_value = None
    proc.pid = 55501

    def _record_grace(*_args: object, **_kwargs: object) -> None:
        calls.append(proc.pid)

    monkeypatch.setattr(
        "shared.python.ai.adapters.cli_subprocess_runner._graceful_then_forced_kill",
        _record_grace,
    )
    exc = CliSubprocessTimeoutError(["cmd"], 1.0)
    assert terminate_cli_subprocess_after_error(exc, proc=proc) is True
    assert calls == [55501]


def test_invoke_timeout_terminates_process_group_with_child_pid(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pathlib import Path

    bin_dir = Path(tmp_path)
    monkeypatch.setenv("TOOLS_FAKE_CLI_MODE", "sleep_tree")
    monkeypatch.setenv("TOOLS_FAKE_CLI_GRANDCHILD_GRACE_SEC", "60")
    binary = install_fake_cli(bin_dir, "claude")
    recorded: list[int | None] = []

    def _record(pid: int | None) -> bool:
        recorded.append(pid)
        return terminate_cli_process_group(pid)

    with patch(
        "shared.python.ai.adapters.cli_subprocess_runner.terminate_cli_process_group",
        side_effect=_record,
    ):
        with pytest.raises(AITimeoutError):
            invoke_cli_subprocess_for_adapter(
                str(binary),
                [str(binary), "-p"],
                provider="claude_code",
                timeout=0.5,
            )
    assert any(isinstance(pid, int) and pid > 0 for pid in recorded)


def test_codex_adapter_timeout_invokes_group_termination(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pathlib import Path

    from shared.python.ai.adapters.codex_cli_adapter import CodexCliAdapter

    bin_dir = Path(tmp_path)
    monkeypatch.setenv("TOOLS_FAKE_CLI_MODE", "sleep_tree")
    monkeypatch.setenv("TOOLS_FAKE_CLI_GRANDCHILD_GRACE_SEC", "60")
    binary = install_fake_cli(bin_dir, "codex")
    recorded: list[int | None] = []

    def _record(pid: int | None) -> bool:
        recorded.append(pid)
        return terminate_cli_process_group(pid)

    with patch(
        "shared.python.ai.adapters.cli_subprocess_runner.terminate_cli_process_group",
        side_effect=_record,
    ):
        adapter = CodexCliAdapter(binary=str(binary), timeout=0.5)
        with pytest.raises(AITimeoutError):
            adapter.send_message("hang", ConversationContext(), [])
    assert any(isinstance(pid, int) and pid > 0 for pid in recorded)
