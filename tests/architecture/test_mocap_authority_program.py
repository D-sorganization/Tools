from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ADR = ROOT / "docs" / "adr" / "ADR-008-markerless-mocap-authority-and-licensing.md"
ACCEPTANCE = ROOT / "docs" / "specs" / "MARKERLESS_MOCAP_ACCEPTANCE_PROGRAM.md"
HANDOFF = (
    ROOT
    / "src"
    / "shared"
    / "python"
    / "sidekick"
    / "lab"
    / "mocap"
    / "AGENT_HANDOFF.md"
)


def test_authority_adr_records_required_cross_repository_boundaries() -> None:
    text = ADR.read_text(encoding="utf-8")
    for phrase in (
        "Tools owns",
        "UpstreamDrift owns",
        "AffineDrift owns",
        "Tools_Private",
        "camera agnostic",
        "FreeMoCap",
        "SkellyCam",
        "AGPL",
        "legal review",
        "T_world_from_camera",
        "x toward target",
        "y up",
        "z right",
        "single-camera",
        "triangulated 3-D",
        "C3D",
    ):
        assert phrase in text


def test_acceptance_program_is_executable_and_fail_closed() -> None:
    text = ACCEPTANCE.read_text(encoding="utf-8")
    for phrase in (
        "Known-geometry synthetic rig",
        "mixed-camera",
        "clock skew",
        "dropped frames",
        "independent C3D reader",
        "exact-HEAD clean export",
        "post-merge",
        "Unavailable",
    ):
        assert phrase in text


def test_mocap_handoff_is_current_and_bounded() -> None:
    lines = HANDOFF.read_text(encoding="utf-8").splitlines()
    assert len(lines) <= 150
    assert "#4708" in "\n".join(lines)
    assert "#4710" in "\n".join(lines)
