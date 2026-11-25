"""Epic #6019: stdio MCP list, describe, and run acceptance for baghouse."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = [pytest.mark.acceptance]


@pytest.fixture
def repo_root() -> Path:
    return _REPO_ROOT


def test_baghouse_stdio_mcp_list_describe_run_and_fail_closed(repo_root: Path) -> None:
    """TDD: red until baghouse_mcp_probe exercises the live stdio MCP server."""
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

    probe = Path(__file__).with_name("baghouse_mcp_probe.py")
    result = subprocess.run(
        [sys.executable, str(probe), str(repo_root)],
        cwd=_REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(_REPO_ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
