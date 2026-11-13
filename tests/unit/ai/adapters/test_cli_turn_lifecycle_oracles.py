"""Tier-1 oracles for CLI turn cancellation, idle timeout, and outcomes (#5727)."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from shared.python.ai.adapters.claude_code_adapter import ClaudeCodeAdapter
from shared.python.ai.adapters.cli_subprocess_runner import (
    run_cli_subprocess,
)
from shared.python.ai.adapters.cli_turn_lifecycle import CliCancelToken
from shared.python.ai.adapters.codex_cli_adapter import CodexCliAdapter
from shared.python.ai.adapters.gemini_cli_adapter import GeminiCliAdapter
from shared.python.ai.exceptions import (
    AIAuthenticationError,
    AICancelledError,
    AIConnectionError,
    AIIdleTimeoutError,
    AIProviderError,
    AIRateLimitError,
    AITimeoutError,
    AIUsageExhaustedError,
)
from shared.python.ai.gui.assistant_widgets import StreamWorker
from shared.python.ai.types import ConversationContext
from shared.python.lifecycle import LifecycleSnapshot, LifecycleStatus
from tests.unit.ai.adapters.test_cli_subprocess_runner_oracles import (
    _proc_alive,
    _wait_proc_dead,
    install_fake_cli,
)

pytestmark = pytest.mark.unit

_FIXTURE_ERR_DIR = Path(__file__).resolve().parents[3] / "fixtures" / "cli_errors"


def test_oracle_02_cancel_live_cli_kills_tree_within_2s(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Oracle 2: cancel() kills tree within 2s and records CANCELLED."""
    monkeypatch.setenv("TOOLS_FAKE_CLI_MODE", "sleep_tree")
    monkeypatch.setenv("TOOLS_FAKE_CLI_GRANDCHILD_GRACE_SEC", "60")
    pid_file = tmp_path / "tree.pids"
    monkeypatch.setenv("TOOLS_FAKE_CLI_TREE_PID_FILE", str(pid_file))
    adapter = ClaudeCodeAdapter(
        binary=str(install_fake_cli(tmp_path, "claude")), timeout=30.0
    )

    cancel_at: list[float] = []

    def _trigger_cancel() -> None:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not pid_file.exists():
            time.sleep(0.05)
        time.sleep(0.1)
        cancel_at.append(time.monotonic())
        adapter.cancel()

    t = threading.Thread(target=_trigger_cancel, daemon=True)
    t.start()
    with pytest.raises(AICancelledError) as exc_info:
        adapter.send_message("sleep", ConversationContext(), [])
    elapsed = time.monotonic() - cancel_at[0]
    t.join(timeout=2.0)

    assert type(exc_info.value) is AICancelledError
    assert elapsed < 2.0
    assert adapter.last_turn_lifecycle is not None
    assert adapter.last_turn_lifecycle.status is LifecycleStatus.CANCELLED
    parent_pid, child_pid = (int(line) for line in pid_file.read_text().splitlines())
    assert not _proc_alive(parent_pid)
    assert not _proc_alive(child_pid)


def test_oracle_03_pre_cancel_double_cancel_and_stream_worker_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Oracle 3: pre-cancel skips spawn; post-exit & double cancel harmless."""
    monkeypatch.setenv("TOOLS_FAKE_CLI_MODE", "sleep_tree")
    pid_file = tmp_path / "pre_cancel.pids"
    monkeypatch.setenv("TOOLS_FAKE_CLI_TREE_PID_FILE", str(pid_file))
    binary = install_fake_cli(tmp_path, "claude")

    token = CliCancelToken()
    token.cancel()
    with pytest.raises(AICancelledError):
        ClaudeCodeAdapter(binary=str(binary)).send_message(
            "hi", ConversationContext(), [], cancel_token=token
        )
    assert not pid_file.exists()

    with pytest.raises(Exception) as run_exc:
        run_cli_subprocess([str(binary)], timeout=5.0, cancel_token=token)
    assert "cancel" in type(run_exc.value).__name__.lower()
    assert not pid_file.exists()

    adapter = ClaudeCodeAdapter(binary=str(binary), timeout=5.0)
    monkeypatch.setenv("TOOLS_FAKE_CLI_MODE", "ok")
    resp = adapter.send_message("ok", ConversationContext(), [])
    assert resp.content == "ok"
    assert adapter.cancel() is False
    assert adapter.cancel() is False

    worker = StreamWorker(MagicMock(), "hi", ConversationContext(), [])
    seen_cancelled: list[bool] = []
    seen_finished: list[bool] = []
    worker.cancelled.connect(lambda: seen_cancelled.append(True))
    worker.finished.connect(lambda: seen_finished.append(True))
    assert worker.cancel() is True
    worker.run()
    assert seen_cancelled == [True]
    assert seen_finished == [True]


def test_oracle_05_parent_exit_kills_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Oracle 5: hard-killing parent process reaps fake CLI and grandchild."""
    env = {
        **os.environ,
        "TOOLS_FAKE_CLI_MODE": "sleep_tree",
        "TOOLS_FAKE_CLI_GRANDCHILD_GRACE_SEC": "60",
    }
    pid_file = tmp_path / "orphan.pids"
    env["TOOLS_FAKE_CLI_TREE_PID_FILE"] = str(pid_file)
    binary = str(install_fake_cli(tmp_path, "claude"))
    helper_code = (
        "from shared.python.ai.adapters.cli_subprocess_runner import "
        "run_cli_subprocess;"
        f"run_cli_subprocess([{binary!r}], timeout=60.0)"
    )
    helper = subprocess.Popen(
        [sys.executable, "-c", helper_code],
        env={**env, "PYTHONPATH": str(Path(__file__).resolve().parents[4] / "src")},
        cwd=str(Path(__file__).resolve().parents[4]),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline and not pid_file.exists():
            time.sleep(0.05)
        assert pid_file.exists(), "fake CLI did not start in time"
        lines = pid_file.read_text(encoding="utf-8").splitlines()
        fake_pid, grandchild_pid = (int(line) for line in lines[:2])
        helper.kill()
        helper.wait(timeout=5.0)
        assert _wait_proc_dead(fake_pid, timeout_sec=2.5)
        assert _wait_proc_dead(grandchild_pid, timeout_sec=2.5)
    finally:
        if helper.poll() is None:
            helper.kill()
            helper.wait(timeout=2.0)


def test_oracle_08_idle_timeout_preserves_partial_while_heartbeat_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Oracle 8: idle stall preserves partial output; heartbeats pass."""
    binary = str(install_fake_cli(tmp_path, "claude"))
    monkeypatch.setenv("TOOLS_FAKE_CLI_MODE", "stall_after_line")
    adapter = ClaudeCodeAdapter(binary=binary, timeout=10.0, idle_timeout=0.4)
    started = time.monotonic()
    with pytest.raises(AIIdleTimeoutError) as exc_info:
        adapter.send_message("stall", ConversationContext(), [])
    elapsed = time.monotonic() - started
    assert elapsed < 3.5
    assert exc_info.value.is_partial is True
    assert "first partial line" in exc_info.value.partial_output
    assert adapter.last_turn_lifecycle is not None
    assert adapter.last_turn_lifecycle.outcome == "idle_timeout"

    monkeypatch.setenv("TOOLS_FAKE_CLI_MODE", "heartbeat_lines")
    monkeypatch.setenv("TOOLS_FAKE_CLI_BEATS", "8")
    steady = ClaudeCodeAdapter(binary=binary, timeout=5.0, idle_timeout=0.5)
    response = steady.send_message("beats", ConversationContext(), [])
    assert "beat 7" in response.content
    assert steady.last_turn_lifecycle is not None
    assert steady.last_turn_lifecycle.status is LifecycleStatus.SUCCEEDED


@pytest.mark.parametrize(
    ("adapter_cls", "bin_name", "fixture_name", "expected_exc"),
    [
        (
            ClaudeCodeAdapter,
            "claude",
            "claude-code-1.0.0_not_logged_in.txt",
            AIAuthenticationError,
        ),
        (
            CodexCliAdapter,
            "codex",
            "codex-cli-0.1.0_rate_limited.txt",
            AIRateLimitError,
        ),
        (
            GeminiCliAdapter,
            "gemini",
            "agy-0.2.0_usage_exhausted.txt",
            AIUsageExhaustedError,
        ),
    ],
)
def test_oracle_11_typed_cli_exit_errors_distinct(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    adapter_cls: type[ClaudeCodeAdapter | CodexCliAdapter | GeminiCliAdapter],
    bin_name: str,
    fixture_name: str,
    expected_exc: type[AIProviderError],
) -> None:
    """Oracle 11: versioned CLI error stderr maps to distinct typed provider errors."""
    classes = (AIAuthenticationError, AIRateLimitError, AIUsageExhaustedError)
    assert len(set(classes)) == 3
    for a in classes:
        for b in classes:
            if a is not b:
                assert not issubclass(a, b)
    monkeypatch.setenv("TOOLS_FAKE_CLI_MODE", "stderr_fixture")
    monkeypatch.setenv(
        "TOOLS_FAKE_CLI_STDERR_FILE", str(_FIXTURE_ERR_DIR / fixture_name)
    )
    monkeypatch.setenv("TOOLS_FAKE_CLI_EXIT_CODE", "1")
    adapter = adapter_cls(binary=str(install_fake_cli(tmp_path, bin_name)), timeout=5.0)
    with pytest.raises(expected_exc):
        adapter.send_message("trigger", ConversationContext(), [])
    assert adapter.last_turn_lifecycle is not None
    assert adapter.last_turn_lifecycle.status is LifecycleStatus.FAILED


@pytest.mark.parametrize(
    ("case_name", "expected_status", "expected_outcome"),
    [
        ("completed", LifecycleStatus.SUCCEEDED, "completed"),
        ("cancelled", LifecycleStatus.CANCELLED, "cancelled"),
        ("timeout", LifecycleStatus.FAILED, "timeout"),
        ("idle_timeout", LifecycleStatus.FAILED, "idle_timeout"),
        ("not_logged_in", LifecycleStatus.FAILED, "not_logged_in"),
        ("rate_limited", LifecycleStatus.FAILED, "rate_limited"),
        ("usage_exhausted", LifecycleStatus.FAILED, "usage_exhausted"),
        ("connection_error", LifecycleStatus.FAILED, "connection_error"),
        ("nonzero_exit", LifecycleStatus.FAILED, "provider_error"),
    ],
)
def test_oracle_15_each_turn_ends_in_single_canonical_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_name: str,
    expected_status: LifecycleStatus,
    expected_outcome: str,
) -> None:
    """Oracle 15: every CLI turn ends in one canonical LifecycleStatus."""
    binary = str(install_fake_cli(tmp_path, "claude"))
    if case_name == "connection_error":
        adapter = ClaudeCodeAdapter(binary=str(tmp_path / "missing-cli"), timeout=2.0)
        adapter.binary = str(tmp_path / "missing-cli")
        with pytest.raises(AIConnectionError):
            adapter.send_message("hi", ConversationContext(), [])
    elif case_name == "cancelled":
        tok = CliCancelToken()
        tok.cancel()
        adapter = ClaudeCodeAdapter(binary=binary, timeout=2.0)
        with pytest.raises(AICancelledError):
            adapter.send_message("hi", ConversationContext(), [], cancel_token=tok)
    elif case_name == "completed":
        monkeypatch.setenv("TOOLS_FAKE_CLI_MODE", "ok")
        adapter = ClaudeCodeAdapter(binary=binary, timeout=5.0)
        adapter.send_message("hi", ConversationContext(), [])
    elif case_name == "timeout":
        monkeypatch.setenv("TOOLS_FAKE_CLI_MODE", "sleep_tree")
        monkeypatch.setenv("TOOLS_FAKE_CLI_GRANDCHILD_GRACE_SEC", "10")
        adapter = ClaudeCodeAdapter(binary=binary, timeout=0.4)
        with pytest.raises(AITimeoutError):
            adapter.send_message("hi", ConversationContext(), [])
    elif case_name == "idle_timeout":
        monkeypatch.setenv("TOOLS_FAKE_CLI_MODE", "stall_after_line")
        adapter = ClaudeCodeAdapter(binary=binary, timeout=5.0, idle_timeout=0.3)
        with pytest.raises(AIIdleTimeoutError):
            adapter.send_message("hi", ConversationContext(), [])
    elif case_name == "nonzero_exit":
        err = tmp_path / "generic_err.txt"
        err.write_text("unexpected internal error\n", encoding="utf-8")
        monkeypatch.setenv("TOOLS_FAKE_CLI_MODE", "stderr_fixture")
        monkeypatch.setenv("TOOLS_FAKE_CLI_STDERR_FILE", str(err))
        adapter = ClaudeCodeAdapter(binary=binary, timeout=5.0)
        with pytest.raises(AIProviderError):
            adapter.send_message("hi", ConversationContext(), [])
    else:
        fixture_map = {
            "not_logged_in": "claude-code-1.0.0_not_logged_in.txt",
            "rate_limited": "codex-cli-0.1.0_rate_limited.txt",
            "usage_exhausted": "agy-0.2.0_usage_exhausted.txt",
        }
        monkeypatch.setenv("TOOLS_FAKE_CLI_MODE", "stderr_fixture")
        monkeypatch.setenv(
            "TOOLS_FAKE_CLI_STDERR_FILE", str(_FIXTURE_ERR_DIR / fixture_map[case_name])
        )
        adapter = ClaudeCodeAdapter(binary=binary, timeout=5.0)
        with pytest.raises(AIProviderError):
            adapter.send_message("hi", ConversationContext(), [])

    snap = adapter.last_turn_lifecycle
    assert isinstance(snap, LifecycleSnapshot)
    assert snap.done is True
    assert snap.status is expected_status
    assert snap.outcome == expected_outcome
    assert sum([snap.ok, snap.cancelled, snap.status is LifecycleStatus.FAILED]) == 1
