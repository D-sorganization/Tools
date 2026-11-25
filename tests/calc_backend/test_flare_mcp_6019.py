"""MCP stdio acceptance for the ``flare`` calculator (Tools epic #6019)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.headless_safe]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROBE = Path(__file__).with_name("flare_mcp_probe.py")


def _mcp_sdk_available() -> bool:
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
    return probe_import.returncode == 0


_SKIP_NO_MCP = not _mcp_sdk_available()


@pytest.mark.skipif(_SKIP_NO_MCP, reason="Optional MCP SDK is not installed")
def test_flare_stdio_list_describe_run_and_fail_closed(repo_root: Path) -> None:
    """Stdio MCP list/describe/run for flare; invalid input yields FAILED (TDD)."""
    result = subprocess.run(
        [sys.executable, str(_PROBE), str(repo_root)],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root / "src")},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
