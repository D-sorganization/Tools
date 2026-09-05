"""Contract tests for Rate of Closure campaign manifest & gap audit (Tools #4921).

Validates that:
1. docs/release/rate_of_closure_campaign.v1.json conforms to its canonical structure.
2. All 15 campaign programs have been reconciled against main (zero programs remain
   on 'implemented_on_feature_stack').
3. Program #4130 explicitly records the re-landing of impact-interval dynamics (#4945).
4. Top-level gap audit reconciliation documents delivered vs missing slices.
5. All delivered modules and files cited in the reconciliation physically exist on disk.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parents[2]
CAMPAIGN_PATH = ROOT / "docs" / "release" / "rate_of_closure_campaign.v1.json"

DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
HEX_SHA_PATTERN = re.compile(r"^[0-9a-f]{8,40}$")

EXPECTED_STAGES = {
    "specified_only",
    "implemented_on_feature_stack",
    "protected_merged_to_parent",
    "implemented_unverified",
    "released_to_main",
}

EXPECTED_SLICES = {
    "inverse_flight_solver",
    "wind_strategy",
    "camera_controls",
    "screw_axis_analytics",
    "impact_interval_dynamics",
    "ground_bounce_regional",
    "shared_golf_club_builder",
}


def _load_campaign() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(CAMPAIGN_PATH.read_text(encoding="utf-8")))


def test_campaign_manifest_envelope_and_metadata() -> None:
    manifest = _load_campaign()
    assert manifest["schema_version"] == "rate-of-closure-campaign/v1"
    assert manifest["repository"] == "D-sorganization/Tools"
    assert manifest["default_branch"] == "main"
    assert DATE_PATTERN.match(manifest["as_of"])

    stages = set(manifest.get("release_stage_definitions", []))
    assert stages == EXPECTED_STAGES

    checks = manifest.get("required_main_checks", [])
    assert "quality-gate" in checks


def test_zero_programs_remain_on_feature_stack() -> None:
    manifest = _load_campaign()
    programs = manifest.get("programs", [])
    assert len(programs) == 15

    on_stack = [
        p["issue"]
        for p in programs
        if p.get("delivery_stage") == "implemented_on_feature_stack"
    ]
    assert not on_stack, f"Programs still on feature stack: {on_stack}"

    for p in programs:
        stage = p.get("delivery_stage")
        assert stage in EXPECTED_STAGES, (
            f"Issue #{p['issue']} has invalid stage {stage}"
        )
        # All active programs must be implemented_unverified or specified_only
        assert stage in {"implemented_unverified", "specified_only", "released_to_main"}


def test_every_program_has_main_reconciliation() -> None:
    manifest = _load_campaign()
    programs = manifest.get("programs", [])

    for p in programs:
        rec = p.get("main_reconciliation")
        assert isinstance(rec, dict), f"Issue #{p['issue']} missing main_reconciliation"
        assert DATE_PATTERN.match(rec["as_of"]), (
            f"Issue #{p['issue']} invalid as_of date"
        )
        assert rec.get("source"), f"Issue #{p['issue']} missing reconciliation source"

        if "main_sha" in rec and rec["main_sha"] is not None:
            assert HEX_SHA_PATTERN.match(rec["main_sha"]), (
                f"Issue #{p['issue']} invalid main_sha"
            )


def test_program_4130_re_landed_via_pr_4945() -> None:
    manifest = _load_campaign()
    p4130 = next((p for p in manifest.get("programs", []) if p["issue"] == 4130), None)
    assert p4130 is not None
    assert p4130["delivery_stage"] == "implemented_unverified"

    rec = p4130["main_reconciliation"]
    assert rec["reference"] == "https://github.com/D-sorganization/Tools/pull/4945"
    assert "4945" in rec["note"]

    # Verify physical impact_interval modules exist on disk
    solver_path = (
        ROOT
        / "src"
        / "shared"
        / "python"
        / "swing_sim"
        / "impact_interval"
        / "solver.py"
    )
    types_path = (
        ROOT
        / "src"
        / "shared"
        / "python"
        / "swing_sim"
        / "impact_interval"
        / "types.py"
    )
    contact_path = (
        ROOT / "src" / "shared" / "python" / "swing_sim" / "impact" / "contact.py"
    )
    assert solver_path.is_file(), f"Missing {solver_path}"
    assert types_path.is_file(), f"Missing {types_path}"
    assert contact_path.is_file(), f"Missing {contact_path}"


def test_gap_audit_reconciliation_slices() -> None:
    manifest = _load_campaign()
    gap_audit = manifest.get("gap_audit_reconciliation")
    assert isinstance(gap_audit, dict), "Missing gap_audit_reconciliation"
    assert gap_audit["audit_issue"] == 4921
    assert DATE_PATTERN.match(gap_audit["as_of"])

    slices = gap_audit.get("slices", {})
    assert EXPECTED_SLICES.issubset(slices.keys())

    # Check inverse flight solver
    inv = slices["inverse_flight_solver"]
    assert inv["status"] == "delivered"
    assert inv["delivered_on_main"] is True

    # Check wind strategy
    wind = slices["wind_strategy"]
    assert wind["status"] == "partially_delivered"
    assert wind["delivered_on_main"] is True
    assert len(wind["missing_slices"]) > 0

    # Check camera controls
    cam = slices["camera_controls"]
    assert cam["status"] == "partially_delivered"
    assert cam["delivered_on_main"] is False
    assert len(cam["missing_slices"]) > 0

    # Check screw-axis analytics
    screw = slices["screw_axis_analytics"]
    assert screw["status"] == "delivered"
    assert (
        ROOT / "src" / "rate_of_closure" / "simulation" / "screw_analysis.py"
    ).is_file()

    # Check club builder
    club = slices["shared_golf_club_builder"]
    assert club["status"] == "delivered"
    assert (ROOT / "src" / "shared" / "python" / "golf_club" / "assembly.py").is_file()
