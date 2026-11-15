"""Unit tests for scripts/run_cli_discovery_probes.py (#5727)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from shared.python.ai.adapters.cli_qualification import DEFAULT_CLI_DISCOVERY_SPECS

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_cli_discovery_probes.py"


@pytest.mark.unit
def test_run_cli_discovery_probes_script_emits_json_report(tmp_path: Path) -> None:
    """Discovery runner writes one JSON object per configured provider."""
    out = tmp_path / "report.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    for key in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
    ):
        env.pop(key, None)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(out)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode in (0, 1), proc.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "tools/cli-discovery-report/v1"
    assert len(payload["probes"]) == len(DEFAULT_CLI_DISCOVERY_SPECS)
    providers = {item["provider"] for item in payload["probes"]}
    assert providers == {spec.provider for spec in DEFAULT_CLI_DISCOVERY_SPECS}


@pytest.mark.unit
def test_run_cli_discovery_probes_rejects_api_key_env(tmp_path: Path) -> None:
    """Forbidden API-key env vars fail closed before probing."""
    out = tmp_path / "reject.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(out)],
        cwd=str(REPO_ROOT),
        env={
            "PYTHONPATH": str(REPO_ROOT / "src"),
            "ANTHROPIC_API_KEY": "forbidden-in-test",  # pragma: allowlist secret
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2, proc.stdout
    assert not out.exists()
