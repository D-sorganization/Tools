"""Fail-closed delivery admission for Tools intrinsic calibration issue 4714."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = (
    REPO_ROOT
    / "docs"
    / "development"
    / "mocap_intrinsic_calibration_4714_reconciliation.json"
)
HANDOFF_PATH = REPO_ROOT / "AGENT_HANDOFF.md"
EXPECTED_BASE = "cff2909f1585273e10fa49165bfab8521e889da1"
EXPECTED_RUNTIME_CANDIDATE = "619b23f27548dbd821b511f27a02b084d9d2ac63"
REQUIRED_DEPENDENCIES = {"TOOLS-M0", "TOOLS-M1", "TOOLS-M2", "TOOLS-M3"}
FORBIDDEN_RUNTIME_PATHS = {
    "src/shared/python/sidekick/lab/mocap/_intrinsic_quality.py",
    "src/shared/python/sidekick/lab/mocap/calibration.py",
    "src/shared/python/sidekick/lab/mocap/calibration_result.py",
    "src/shared/python/sidekick/lab/mocap/intrinsic_solver.py",
}


def _ledger() -> dict[str, object]:
    return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))


def test_intrinsic_delivery_remains_blocked_on_unreleased_authority() -> None:
    ledger = _ledger()

    assert ledger["schema_version"] == "mocap-intrinsic-reconciliation/1.0.0"
    assert ledger["authority_repository"] == "D-sorganization/Tools"
    assert ledger["protected_main_sha"] == EXPECTED_BASE
    assert ledger["prior_runtime_candidate"]["head_sha"] == EXPECTED_RUNTIME_CANDIDATE
    assert ledger["decision"] == "blocked_dependency_reconciliation"
    assert ledger["runtime_delivery_authorized"] is False
    assert ledger["pull_request_authorized"] is False
    assert ledger["adoption_claim"] == "not_implemented"


def test_all_m0_through_m3_dependencies_are_explicit_and_unreleased() -> None:
    dependencies = _ledger()["dependencies"]
    by_id = {item["id"]: item for item in dependencies}

    assert set(by_id) == REQUIRED_DEPENDENCIES
    assert all(item["release_state"] == "not_merged" for item in by_id.values())
    assert by_id["TOOLS-M0"]["provider_pr"] == 4734
    assert by_id["TOOLS-M1"]["provider_pr"] == 4734
    assert by_id["TOOLS-M2"]["provider_pr"] is None
    assert by_id["TOOLS-M3"]["provider_pr"] is None


def test_blocked_audit_branch_cannot_publish_calibration_runtime() -> None:
    ledger = _ledger()

    assert set(ledger["prohibited_runtime_paths"]) == FORBIDDEN_RUNTIME_PATHS
    assert all(not (REPO_ROOT / path).exists() for path in FORBIDDEN_RUNTIME_PATHS)
    assert ledger["prior_runtime_candidate"]["release_authority"] is False
    assert ledger["prior_runtime_candidate"]["safe_to_cherry_pick"] is False


def test_reconciliation_plan_preserves_provider_and_consumer_boundaries() -> None:
    ledger = _ledger()
    sequence = ledger["required_sequence"]
    sequence_ids = [item["id"] for item in sequence]

    assert sequence_ids == ["R1", "R2", "R3", "R4", "R5", "R6"]
    assert ledger["consumer"]["repository"] == "D-sorganization/AffineDrift"
    assert ledger["consumer"]["issue"] == 3962
    assert ledger["consumer"]["tools_runtime_state"] == "unavailable"
    assert ledger["planned_m4"]["pattern_provider_depends_on"] == [
        "TOOLS-M2",
        "TOOLS-M3",
    ]
    assert ledger["planned_m4"]["physical_qualification"] == "not_claimed"


def test_root_handoff_names_the_fail_closed_issue_state() -> None:
    handoff = HANDOFF_PATH.read_text(encoding="utf-8")

    assert "#4714" in handoff
    assert "blocked_dependency_reconciliation" in handoff
    assert EXPECTED_RUNTIME_CANDIDATE in handoff
