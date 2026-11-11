"""Exercise ud-tools-mcp with the real MCP SDK."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_real_mcp_lists_and_runs_three_calculators(repo_root: Path) -> None:
    probe_import = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from mcp.server.fastmcp import FastMCP; "
                "from mcp.client.stdio import stdio_client"
            ),
        ],
        capture_output=True,
        check=False,
    )
    if probe_import.returncode != 0:
        pytest.skip("Optional MCP SDK with FastMCP is not installed or incompatible")

    probe = Path(__file__).with_name("mcp_probe.py")
    result = subprocess.run(
        [sys.executable, str(probe), str(repo_root)],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root / "src")},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
