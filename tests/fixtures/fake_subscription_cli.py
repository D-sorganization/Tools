#!/usr/bin/env python3
"""Tier-1 fake ``claude`` / ``codex`` / ``agy`` executable for adapter process tests."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_MODE_ENV = "TOOLS_FAKE_CLI_MODE"
_PROMPT_FILE_ENV = "TOOLS_FAKE_CLI_PROMPT_FILE"
_SIDEKICK_PROMPT_FILE_ENV = "SIDEKICK_CLI_PROMPT_FILE"
_TREE_PID_ENV = "TOOLS_FAKE_CLI_TREE_PID_FILE"
_GRANDCHILD_GRACE_SEC = "TOOLS_FAKE_CLI_GRANDCHILD_GRACE_SEC"
_OUTPUT_MB = "TOOLS_FAKE_CLI_OUTPUT_MB"
_VERSION_ENV = "TOOLS_FAKE_CLI_VERSION"
_ENCODING_SAMPLE_ENV = "TOOLS_FAKE_CLI_ENCODING_SAMPLE"


def _read_prompt() -> str:
    for key in (_SIDEKICK_PROMPT_FILE_ENV, _PROMPT_FILE_ENV):
        if path := os.environ.get(key):
            return Path(path).read_text(encoding="utf-8")
    for index, arg in enumerate(sys.argv):
        if arg in {"-p", "--prompt"} and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
    if len(sys.argv) > 1 and sys.argv[-1] not in {
        "--skip-trust",
        "--skip-git-repo-check",
        "--version",
    }:
        return sys.argv[-1]
    if not sys.stdin.isatty():
        payload = sys.stdin.read()
        if payload:
            return payload
    return ""


def _mode_sleep_tree() -> int:
    pid_file = os.environ.get(_TREE_PID_ENV)
    grace = float(os.environ.get(_GRANDCHILD_GRACE_SEC, "60"))
    grandchild = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(3600)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if pid_file:
        Path(pid_file).write_text(
            f"{os.getpid()}\n{grandchild.pid}\n",
            encoding="utf-8",
        )
    time.sleep(grace)
    return 0


def _mode_output_before_stdin() -> int:
    """Fill stdout before stdin; parent deadlocks unless streams drain first."""
    mebibytes = int(os.environ.get(_OUTPUT_MB, "5"))
    chunk = b"y" * (1024 * 1024)
    for _ in range(mebibytes):
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
    if not sys.stdin.isatty():
        payload = sys.stdin.buffer.read()
        sys.stderr.buffer.write(payload)
        sys.stderr.buffer.flush()
    return 0


def _mode_large_output() -> int:
    mebibytes = int(os.environ.get(_OUTPUT_MB, "5"))
    chunk = b"x" * (1024 * 1024)
    for _ in range(mebibytes):
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
        sys.stderr.buffer.write(chunk)
        sys.stderr.buffer.flush()
    return 0


def _mode_stdin_eof() -> int:
    if sys.stdin.isatty():
        return 1
    payload = sys.stdin.buffer.read()
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()
    return 0


def _mode_echo_prompt() -> int:
    prompt = _read_prompt()
    sys.stdout.write(prompt)
    sys.stdout.flush()
    return 0


def _mode_trap_term() -> int:
    ignored = {"trapped": False}

    def _on_term(signum: int, _frame: object) -> None:
        ignored["trapped"] = True

    signal.signal(signal.SIGTERM, _on_term)
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline and not ignored["trapped"]:
        time.sleep(0.05)
    return 0


def _mode_encoding_roundtrip() -> int:
    if sample := os.environ.get(_ENCODING_SAMPLE_ENV):
        sys.stdout.buffer.write(bytes.fromhex(sample))
        sys.stdout.buffer.flush()
    prompt = _read_prompt()
    sys.stdout.write(prompt)
    sys.stdout.flush()
    return 0


def _dispatch() -> int:
    if "--version" in sys.argv:
        print(os.environ.get(_VERSION_ENV, "fake-cli 0.0.0-test"))
        return 0
    mode = os.environ.get(_MODE_ENV, "ok").strip().lower()
    if mode == "ok":
        sys.stdout.write("ok\n")
        return 0
    if mode == "sleep_tree":
        return _mode_sleep_tree()
    if mode == "large_output":
        return _mode_large_output()
    if mode == "output_before_stdin":
        return _mode_output_before_stdin()
    if mode == "stdin_eof":
        return _mode_stdin_eof()
    if mode == "echo_prompt":
        return _mode_echo_prompt()
    if mode == "trap_term":
        return _mode_trap_term()
    if mode == "encoding_roundtrip":
        return _mode_encoding_roundtrip()
    sys.stderr.write(f"unknown fake cli mode: {mode}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(_dispatch())
