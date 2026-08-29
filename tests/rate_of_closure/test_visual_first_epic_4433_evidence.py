"""Fail-closed acceptance evidence for visual-first epic #4433."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "docs/audits/rate_of_closure_visual_first_epic_4433.v1.json"
LEDGER = ROOT / "docs/audits/rate_of_closure_epic_4142_evidence.v1.json"
EXTENSION_GUIDE = (
    ROOT / "docs/development/rate-visualization-extension-and-consumer-guide.md"
)

EXPECTED_REQUIREMENTS = {
    *(f"V0.{index}" for index in range(1, 5)),
    *(f"V1.{index}" for index in range(1, 6)),
    *(f"V2.{index}" for index in range(1, 6)),
    *(f"V3.{index}" for index in range(1, 6)),
    *(f"V4.{index}" for index in range(1, 7)),
    *(f"V5.{index}" for index in range(1, 5)),
    "AM.1",
    "C.1",
}
ALLOWED_STATUSES = {"verified", "partial", "unverified"}


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_visual_first_epic_acceptance_is_exhaustive_and_fail_closed() -> None:
    audit = _load(AUDIT)
    assert audit["schema_id"] == "rate-of-closure/visual-first-epic-evidence"
    assert audit["schema_version"] == 1
    assert audit["epic_url"] == "https://github.com/D-sorganization/Tools/issues/4433"
    assert audit["overall_status"] == "partial"
    assert len(audit["audited_main_commit"]) == 40
    assert audit["protected_run"] == (
        "https://github.com/D-sorganization/Tools/actions/runs/32689177846"
    )

    requirements = audit["requirements"]
    assert isinstance(requirements, list)
    indexed = {item["requirement_id"]: item for item in requirements}
    assert set(indexed) == EXPECTED_REQUIREMENTS
    assert len(indexed) == len(requirements)

    observed = Counter(item["status"] for item in requirements)
    assert dict(observed) == audit["status_counts"]
    for item in requirements:
        assert item["status"] in ALLOWED_STATUSES
        assert item["requirement"].strip()
        assert item["rationale"].strip()
        assert item["evidence_files"]
        for relative in item["evidence_files"]:
            assert "*" not in relative
            assert (ROOT / relative).is_file(), relative
        if item["status"] == "verified":
            assert item["gaps"] == []
        else:
            assert item["gaps"]

    assert audit["external_human_actions"] == [
        "Execute and sign the manual assistive-technology protocol for both surfaces.",
        "Approve the representative rendered review with exact build and image "
        "identities.",
    ]

    v0_1 = indexed["V0.1"]
    assert v0_1["status"] == "verified"
    assert v0_1["gaps"] == []
    assert "src/rate_of_closure/visualization_tab_manifest.py" in v0_1["evidence_files"]
    assert audit["status_counts"] == {"verified": 10, "partial": 21}

    v5_2 = indexed["V5.2"]
    assert v5_2["status"] == "verified"
    assert v5_2["gaps"] == []
    assert "scripts/check_rate_visual_evidence_changes.py" in v5_2["evidence_files"]
    assert ".github/workflows/rate-web-playwright.yml" in v5_2["evidence_files"]

    for requirement_id in ("V5.1", "V5.4"):
        requirement = indexed[requirement_id]
        assert requirement["status"] == "verified"
        assert requirement["gaps"] == []
        assert (
            "docs/development/rate-visualization-extension-and-consumer-guide.md"
            in requirement["evidence_files"]
        )


def test_extension_guide_freezes_lockstep_commands_and_ownership_boundaries() -> None:
    guide = EXTENSION_GUIDE.read_text(encoding="utf-8")
    for path in (
        "visualization_tabs.v1.json",
        "visualization_accessibility.v1.json",
        "visualization_performance.v1.json",
        "visual_baselines.v1.json",
        "visualization_acceptance.v1.json",
    ):
        assert path in guide
    for issue in (
        "D-sorganization/Tools/issues/4135",
        "D-sorganization/Tools/issues/4142",
        "D-sorganization/Tools/issues/4144",
        "D-sorganization/Tools/issues/4433",
        "D-sorganization/Tools/issues/4832",
        "D-sorganization/UpstreamDrift/issues/8358",
        "D-sorganization/UpstreamDrift/pull/9134",
    ):
        assert issue in guide
    assert "d7a95e2a4024f0f3c1d18f9790143cd766032cd3" in guide
    assert "f3832b04454c97a7a0999906972f394474107dd7" in guide
    assert "python -m pytest -n 0" in guide
    assert "npm run test:e2e" in guide
    assert "Tools owns" in guide
    assert "UpstreamDrift owns" in guide
    assert "does not" in guide.lower()


def test_r14_6_retains_the_visual_first_audit_as_partial_evidence() -> None:
    audit = _load(AUDIT)
    ledger = _load(LEDGER)
    r14_6 = next(
        item for item in ledger["requirements"] if item["requirement_id"] == "R14.6"
    )
    relative = "docs/audits/rate_of_closure_visual_first_epic_4433.v1.json"
    assert r14_6["status"] == "partial"
    assert relative in r14_6["evidence_files"]
    assert r14_6["gaps"] == audit["r14_6_blocking_gaps"]
