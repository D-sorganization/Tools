"""Exercise the real optional MCP SDK outside pytest's unrelated mcp namespace."""

from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_real_mcp_lists_only_retrieval_tools_and_observes_edits(
    repository: Path,
) -> None:
    try:
        importlib.metadata.version("mcp")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("Optional MCP SDK is not installed")
    source = Path(__file__).resolve().parents[3] / "src"
    probe = Path(__file__).with_name("mcp_probe.py")
    result = subprocess.run(
        [sys.executable, str(probe), str(repository)],
        cwd=repository,
        env={**os.environ, "PYTHONPATH": str(source)},
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
