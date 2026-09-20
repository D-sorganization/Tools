"""Tests for Rate-of-Closure release gate runner and validators (Tools #4201, #4922)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.release_gate import (
    check_companion_and_playwright,
    check_documentation_and_a11y,
    check_frozen_pyqt_qualification,
    check_parity_inventory,
    check_sbom_and_package_assets,
    main,
    run_release_gate,
    update_campaign_manifest,
)

_REPO_ROOT = Path(__file__).parents[2].resolve()
_SRC_DIR = str((_REPO_ROOT / "src").resolve())
if _SRC_DIR in sys.path:
    sys.path.remove(_SRC_DIR)
sys.path.insert(0, _SRC_DIR)

_ROC_DIR = str((_REPO_ROOT / "src" / "rate_of_closure").resolve())
_roc = sys.modules.get("rate_of_closure")
if _roc is not None:
    _paths = list(getattr(_roc, "__path__", []))
    if _ROC_DIR not in _paths:
        _roc.__path__ = [_ROC_DIR] + _paths


def test_parity_inventory_check() -> None:
    res = check_parity_inventory(_REPO_ROOT)
    assert res["status"] == "passed"
    golden = "regional_ground_execution_result_golden_v1.json"
    assert golden in res["required_golden_verified"]


def test_companion_and_playwright_check() -> None:
    res = check_companion_and_playwright(_REPO_ROOT)
    assert res["status"] == "passed"
    assert res["companion_modules_verified"] >= 4
    assert res["playwright_specs_verified"] >= 4


def test_frozen_pyqt_qualification_offline() -> None:
    res = check_frozen_pyqt_qualification(_REPO_ROOT, skip_binary=True)
    assert res["status"] == "passed"
    assert res["spec_file"] == "rate_of_closure.spec"
    caps = res["capabilities_probe"]
    assert caps["direct_worker_restart_recovery"]["supported"] is False
    assert "backend" in caps["rust_capability"]


def test_sbom_and_package_assets_check() -> None:
    res = check_sbom_and_package_assets(_REPO_ROOT)
    assert res["status"] == "passed"
    assert res["pyproject_verified"] is True
    assert "example_driver_head.stl" in res["assets_verified"]


def test_documentation_and_a11y_check() -> None:
    res = check_documentation_and_a11y(_REPO_ROOT)
    assert res["status"] == "passed"
    assert res["documentation_files_verified"] >= 6
    assert res["accessibility_tabs_verified"] == 20


def test_run_release_gate_offline() -> None:
    report = run_release_gate(_REPO_ROOT, skip_frozen_binary=True)
    assert report["release_gate"] == "RateOfClosure"
    assert report["status"] == "passed"
    assert report["evidence_id"] == "release-gate-verified-4922"
    pillars = report["pillars"]
    assert len(pillars) == 5
    for p in pillars.values():
        assert p["status"] == "passed"


def test_cli_runner_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--repo-root", str(_REPO_ROOT), "--skip-frozen-binary", "--json"])
    assert code == 0
    captured = capsys.readouterr()
    doc = json.loads(captured.out)
    assert doc["status"] == "passed"
    assert doc["release_gate"] == "RateOfClosure"


def test_cli_runner_text_output(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--repo-root", str(_REPO_ROOT), "--skip-frozen-binary"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Rate-of-Closure Release Gate: PASSED" in captured.out
    assert "Evidence ID: release-gate-verified-4922" in captured.out


def test_update_campaign_manifest_in_temp_repo(tmp_path: Path) -> None:
    # Set up minimal temporary repo structure
    docs_dir = tmp_path / "docs" / "release"
    docs_dir.mkdir(parents=True, exist_ok=True)
    campaign_file = docs_dir / "rate_of_closure_campaign.v1.json"

    initial_data: dict[str, Any] = {
        "schema_version": "rate-of-closure-campaign/v1",
        "as_of": "2026-09-01",
        "release_stage_definitions": ["implemented_unverified", "released_to_main"],
        "campaign_release": {"status": "not_released"},
        "test_evidence": [],
        "programs": [
            {
                "issue": 4103,
                "title": "Simulation Platform",
                "delivery_stage": "implemented_unverified",
                "evidence_ids": ["prev-ev"],
            },
            {
                "issue": 4201,
                "title": "Release Gate",
                "delivery_stage": "specified_only",
                "evidence_ids": [],
            },
        ],
    }
    campaign_file.write_text(json.dumps(initial_data, indent=2), encoding="utf-8")

    update_campaign_manifest(tmp_path, evidence_id="custom-ev-123")

    updated = json.loads(campaign_file.read_text(encoding="utf-8"))
    assert "verified" in updated["release_stage_definitions"]
    assert updated["campaign_release"]["status"] == "verified_ready_for_release"

    evidence_ids = [e["id"] for e in updated["test_evidence"]]
    assert "custom-ev-123" in evidence_ids

    p4103 = next(p for p in updated["programs"] if p["issue"] == 4103)
    assert p4103["delivery_stage"] == "verified"
    assert "custom-ev-123" in p4103["evidence_ids"]

    p4201 = next(p for p in updated["programs"] if p["issue"] == 4201)
    assert p4201["delivery_stage"] == "verified"
    assert "custom-ev-123" in p4201["evidence_ids"]
    assert p4201["release"]["status"] == "verified_ready_for_release"
