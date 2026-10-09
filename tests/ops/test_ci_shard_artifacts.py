"""Tests for run-attempt-bound CI shard artifact contracts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_SCRIPT = REPO_ROOT / "scripts" / "ci_shard_artifacts.py"
pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _load_artifacts_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "ci_shard_artifacts", ARTIFACTS_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_status_manifest_binds_exact_workflow_attempt_and_content(
    tmp_path: Path,
) -> None:
    artifacts = _load_artifacts_module()
    run = artifacts.ShardRun("37956815685", 10, "3.12")

    path = artifacts.write_status_manifest(tmp_path, run, "src-shared", "success")
    payload = json.loads(path.read_text(encoding="utf-8"))
    content = {key: value for key, value in payload.items() if key != "payload_sha256"}
    expected_digest = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    assert payload["run_id"] == "37956815685"
    assert payload["run_attempt"] == 10
    assert payload["artifact_name"] == (
        "shard-status-37956815685-attempt-10-3.12-src-shared"
    )
    assert payload["payload_sha256"] == expected_digest
