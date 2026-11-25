"""MCP stdio acceptance for the ``thermal-profile`` calculator (Tools epic #6019)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROBE = Path(__file__).with_name("thermal_profile_mcp_probe.py")

pytestmark = [pytest.mark.acceptance, pytest.mark.headless_safe]


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


def _run_probe(*extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_PROBE), str(_REPO_ROOT), *extra],
        cwd=_REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(_REPO_ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.fixture
def repo_root() -> Path:
    return _REPO_ROOT


@pytest.mark.skipif(_SKIP_NO_MCP, reason="Optional MCP SDK is not installed")
def test_thermal_profile_stdio_list_describe_run(repo_root: Path) -> None:
    """TDD: red until thermal_profile_mcp_probe exercises list/describe/run (#6019)."""
    result = _run_probe()
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(_SKIP_NO_MCP, reason="Optional MCP SDK is not installed")
@pytest.mark.xfail(
    reason=(
        "Tools #6019: model_validator ValueError in allowed_range is not "
        "JSON-serializable over stdio MCP (Unable to serialize unknown type)"
    ),
    strict=False,
)
def test_thermal_profile_stdio_fail_closed_invalid_input(repo_root: Path) -> None:
    """Invalid t_end_s should yield FAILED; stdio serialize defect tracked on #6019."""
    result = _run_probe("fail_closed")
    assert result.returncode == 0, result.stdout + result.stderr
