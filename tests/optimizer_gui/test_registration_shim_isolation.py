"""The shim must resolve after the chat tests are collected (#5878)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_shim_resolves_after_chat_history_browser_collection() -> None:
    # A test that puts tests/ on sys.path makes tests/optimizer_gui shadow the
    # real src/optimizer_gui package for every later import in the process.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:randomly",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            "tests/unit/chat/test_chat_dock_widget_history_browser.py",
            "tests/optimizer_gui/test_registration_shim.py",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
