"""Fail-closed delivery admission for Tools extrinsic calibration issue 4721."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = (
    REPO_ROOT
    / "docs"
    / "development"
    / "mocap_extrinsic_layout_4721_reconciliation.json"
)
HANDOFF_PATH = REPO_ROOT / "AGENT_HANDOFF.md"
EXPECTED_BASE = "cff2909f1585273e10fa49165bfab8521e889da1"
EXPECTED_RUNTIME_CANDIDATE = "f371ce4f06ba9b904d1719a99221cffab44b4020"
REQUIRED_DEPENDENCIES = {
    "TOOLS-M0",
    "TOOLS-M1",
    "TOOLS-M2",
    "TOOLS-M3",
    "TOOLS-M4",
}
REQUIRED_CONTRACT_GAPS = {
    "frames",
    "gauge",
    "robust_estimation",
    "uncertainty",
    "motion_rejection",
    "recalibration",
    "provenance",
}
FORBIDDEN_RUNTIME_PATHS = {
    "src/shared/python/sidekick/lab/mocap/_extrinsic_numerics.py",
    "src/shared/python/sidekick/lab/mocap/_validation.py",
    "src/shared/python/sidekick/lab/mocap/calibration.py",
    "src/shared/python/sidekick/lab/mocap/extrinsic.py",
    "src/shared/python/sidekick/lab/mocap/extrinsic_solver.py",
    "src/shared/python/sidekick/lab/mocap/layout_monitor.py",
}


def _ledger() -> dict[str, object]:
    return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))


def test_extrinsic_delivery_remains_blocked_on_unreleased_authority() -> None:
    ledger = _ledger()

    assert ledger["schema_version"] == "mocap-extrinsic-reconciliation/1.0.0"
    assert ledger["authority_repository"] == "D-sorganization/Tools"
    assert ledger["protected_main_sha"] == EXPECTED_BASE
    assert ledger["prior_runtime_candidate"]["head_sha"] == (
        EXPECTED_RUNTIME_CANDIDATE
    )
    assert ledger["decision"] == "blocked_dependency_reconciliation"
    assert ledger["runtime_delivery_authorized"] is False
    assert ledger["pull_request_authorized"] is False
    assert ledger["adoption_claim"] == "not_implemented"


def test_all_m0_through_m4_dependencies_are_explicit_and_unreleased() -> None:
    dependencies = _ledger()["dependencies"]
    by_id = {item["id"]: item for item in dependencies}

    assert set(by_id) == REQUIRED_DEPENDENCIES
    assert all(item["release_state"] == "not_merged" for item in by_id.values())
    assert by_id["TOOLS-M0"]["provider_pr"] == 4734
    assert by_id["TOOLS-M1"]["provider_pr"] == 4734
    assert by_id["TOOLS-M2"]["provider_pr"] is None
    assert by_id["TOOLS-M3"]["provider_pr"] is None
    assert by_id["TOOLS-M4"]["runtime_state"] == "unavailable"


def test_blocked_audit_branch_cannot_publish_extrinsic_runtime() -> None:
    ledger = _ledger()

    assert set(ledger["prohibited_runtime_paths"]) == FORBIDDEN_RUNTIME_PATHS
    assert all(not (REPO_ROOT / path).exists() for path in FORBIDDEN_RUNTIME_PATHS)
    assert ledger["prior_runtime_candidate"]["release_authority"] is False
    assert ledger["prior_runtime_candidate"]["safe_to_cherry_pick"] is False
    assert ledger["prior_runtime_candidate"]["global_parameter_coupling"] is False


def test_reconciliation_plan_preserves_layout_and_consumer_boundaries() -> None:
    ledger = _ledger()
    sequence = ledger["required_sequence"]

    assert [item["id"] for item in sequence] == [
        "R1",
        "R2",
        "R3",
        "R4",
        "R5",
        "R6",
        "R7",
    ]
    assert set(ledger["planned_m5"]["missing_runtime_contracts"]) == (
        REQUIRED_CONTRACT_GAPS
    )
    assert ledger["planned_m5"]["observation_authority_depends_on"] == [
        "TOOLS-M2",
        "TOOLS-M3",
        "TOOLS-M4",
    ]
    assert ledger["planned_m5"]["physical_qualification"] == "not_claimed"
    assert ledger["consumer"]["repository"] == "D-sorganization/AffineDrift"
    assert ledger["consumer"]["issue"] == 3962
    assert ledger["consumer"]["tools_runtime_state"] == "unavailable"


def test_root_handoff_names_the_fail_closed_issue_state() -> None:
    handoff = HANDOFF_PATH.read_text(encoding="utf-8")

    assert "#4721" in handoff
    assert "blocked_dependency_reconciliation" in handoff
    assert EXPECTED_RUNTIME_CANDIDATE in handoff
