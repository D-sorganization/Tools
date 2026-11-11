"""Tier-1 process oracles for the shared CLI subprocess runner (#5727 wave 1.2)."""

from __future__ import annotations

import os
import stat
import sys
import time
from pathlib import Path

import pytest

from src.shared.python.ai.adapters.claude_code_adapter import ClaudeCodeAdapter
from src.shared.python.ai.adapters.cli_subprocess_runner import (
    DEFAULT_OUTPUT_BYTE_CAP,
    CliSubprocessTimeoutError,
    PromptDelivery,
    run_cli_subprocess,
)
from src.shared.python.ai.adapters.codex_cli_adapter import CodexCliAdapter
from src.shared.python.ai.adapters.gemini_cli_adapter import GeminiCliAdapter
from src.shared.python.ai.exceptions import AIConnectionError, AITimeoutError
from src.shared.python.ai.types import ConversationContext

pytestmark = pytest.mark.unit

_FAKE_SCRIPT = (
    Path(__file__).resolve().parents[3] / "fixtures" / "fake_subscription_cli.py"
)


def install_fake_cli(tmp_path: Path, name: str = "claude") -> Path:
    """Install the fake CLI on ``tmp_path`` and return the executable path."""
    if sys.platform == "win32":
        dest = tmp_path / f"{name}.cmd"
        dest.write_text(
            f'@echo off\r\n"{sys.executable}" "{_FAKE_SCRIPT}" %*\r\n',
            encoding="utf-8",
        )
        return dest
    dest = tmp_path / name
    dest.write_text(
        f'#!/bin/sh\nexec "{sys.executable}" "{_FAKE_SCRIPT}" "$@"\n',
        encoding="utf-8",
    )
    dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return dest


def _proc_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


@pytest.fixture
def fake_bin_dir(tmp_path: Path) -> Path:
    for name in ("claude", "codex", "agy", "gemini"):
        install_fake_cli(tmp_path, name)
    return tmp_path


def test_oracle_01_timeout_kills_grandchild_tree(
    fake_bin_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Oracle 1: timeout kills the fake process and its grandchild."""
    monkeypatch.setenv("TOOLS_FAKE_CLI_MODE", "sleep_tree")
    monkeypatch.setenv("TOOLS_FAKE_CLI_GRANDCHILD_GRACE_SEC", "60")
    pid_file = tmp_path / "tree.pids"
    monkeypatch.setenv("TOOLS_FAKE_CLI_TREE_PID_FILE", str(pid_file))
    binary = install_fake_cli(tmp_path, "claude")
    adapter = ClaudeCodeAdapter(binary=str(binary), timeout=1.0)
    with pytest.raises(AITimeoutError):
        adapter.send_message("hang", ConversationContext(), [])
    assert pid_file.exists()
    parent_pid, child_pid = (int(line) for line in pid_file.read_text().splitlines())
    assert not _proc_alive(parent_pid)
    assert not _proc_alive(child_pid)


def test_oracle_04_graceful_then_forced(
    fake_bin_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Oracle 4: a SIGTERM-trapping fake is still reaped after the grace window."""
    if sys.platform == "win32":
        pytest.skip("POSIX signal trap oracle")
    monkeypatch.setenv("TOOLS_FAKE_CLI_MODE", "trap_term")
    binary = install_fake_cli(fake_bin_dir, "codex")
    with pytest.raises(CliSubprocessTimeoutError):
        run_cli_subprocess(
            [str(binary)],
            timeout=0.5,
            grace_kill_seconds=0.2,
            output_byte_cap=1024,
        )


def test_oracle_06_large_output_no_deadlock(
    fake_bin_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Oracle 6: 5 MiB per stream completes without deadlock; output is bounded."""
    monkeypatch.setenv("TOOLS_FAKE_CLI_MODE", "large_output")
    monkeypatch.setenv("TOOLS_FAKE_CLI_OUTPUT_MB", "5")
    binary = install_fake_cli(fake_bin_dir, "claude")
    result = run_cli_subprocess(
        [str(binary)],
        timeout=30.0,
        output_byte_cap=DEFAULT_OUTPUT_BYTE_CAP,
    )
    assert result.returncode == 0
    assert result.stdout_truncated_bytes > 0
    assert result.stderr_truncated_bytes > 0
    assert "[truncated " in result.stdout
    assert "[truncated " in result.stderr


def test_oracle_07_stdin_closed_codex_and_agy(
    fake_bin_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Oracle 7: codex and agy prompts via stdin return promptly."""
    monkeypatch.setenv("TOOLS_FAKE_CLI_MODE", "stdin_eof")
    codex = install_fake_cli(fake_bin_dir, "codex")
    agy = install_fake_cli(fake_bin_dir, "agy")
    for binary, delivery in (
        (codex, PromptDelivery.STDIN),
        (agy, PromptDelivery.STDIN),
    ):
        started = time.monotonic()
        result = run_cli_subprocess(
            [str(binary)],
            prompt="ping",
            prompt_delivery=delivery,
            timeout=5.0,
        )
        assert time.monotonic() - started < 4.0
        assert result.stdout == "ping"


def test_oracle_09_long_prompt_via_stdin(
    fake_bin_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Oracle 9: a 100k-character prompt reaches the fake intact."""
    monkeypatch.setenv("TOOLS_FAKE_CLI_MODE", "echo_prompt")
    prompt = "P" * 100_000
    codex = CodexCliAdapter(binary=str(install_fake_cli(fake_bin_dir, "codex")))
    response = codex.send_message(prompt, ConversationContext(), [])
    assert prompt in response.content


def test_oracle_10_encoding_roundtrip(
    fake_bin_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Oracle 10: non-UTF-8 bytes and shell-hostile prompt text round-trip."""
    monkeypatch.setenv("TOOLS_FAKE_CLI_MODE", "encoding_roundtrip")
    monkeypatch.setenv("TOOLS_FAKE_CLI_ENCODING_SAMPLE", "cafe")
    prompt = 'quote " percent % caret ^ amp &\nnewline'
    claude = ClaudeCodeAdapter(binary=str(install_fake_cli(fake_bin_dir, "claude")))
    response = claude.send_message(prompt, ConversationContext(), [])
    assert prompt in response.content


def test_oracle_12_missing_and_non_executable_binary(tmp_path: Path) -> None:
    """Oracle 12: missing and non-executable paths become connection errors."""
    adapter = ClaudeCodeAdapter()
    adapter.binary = str(tmp_path / "missing-claude")
    with pytest.raises(AIConnectionError, match="missing-claude"):
        adapter.send_message("hi", ConversationContext(), [])
    bad = tmp_path / "claude"
    bad.write_text("not executable\n", encoding="utf-8")
    adapter = ClaudeCodeAdapter()
    adapter.binary = str(bad)
    with pytest.raises(AIConnectionError, match=str(bad)):
        adapter.send_message("hi", ConversationContext(), [])


def test_gemini_adapter_uses_runner(
    fake_bin_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TOOLS_FAKE_CLI_MODE", "echo_prompt")
    gemini = GeminiCliAdapter(binary=str(install_fake_cli(fake_bin_dir, "gemini")))
    response = gemini.send_message("short", ConversationContext(), [])
    assert "short" in response.content
