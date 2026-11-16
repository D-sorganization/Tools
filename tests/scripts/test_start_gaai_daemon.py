"""The GAAI daemon launcher is retired and must not return (RM #2725)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_start_gaai_daemon_launcher_is_removed() -> None:
    assert not (_REPO_ROOT / "start-gaai-daemon.sh").exists()
    assert not (_REPO_ROOT / ".gaai").exists()
