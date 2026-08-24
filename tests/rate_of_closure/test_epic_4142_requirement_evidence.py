"""Contracts for the requirement-level evidence authority for Tools #4142."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/audits/rate_of_closure_epic_4142_evidence.v1.json"
EXPECTED_REQUIREMENTS = tuple(
    [f"R10.{index}" for index in range(1, 7)]
    + [f"R11.{index}" for index in range(1, 6)]
    + [f"R12.{index}" for index in range(1, 6)]
    + [f"R13.{index}" for index in range(1, 6)]
    + [f"R14.{index}" for index in range(1, 7)]
    + [f"R15.{index}" for index in range(1, 5)]
)
ALLOWED_STATUSES = {"verified", "partial", "unverified", "external_blocked"}
NONAUTHORITATIVE_PREFIXES = (
    "AGENT_HANDOFF.md",
    "SPEC.md",
    "docs/agent_handoff_archive/",
)


def _load() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(EVIDENCE.read_text(encoding="utf-8")))


def test_epic_4142_evidence_covers_every_requirement_exactly_once() -> None:
    evidence = _load()

    assert evidence["schema_version"] == "tools-epic-requirement-evidence/v1"
    assert evidence["repository"] == "D-sorganization/Tools"
    assert evidence["epic"] == 4142
    assert evidence["audit_base_revision"] == (
        "eebdddf8c6e366722be40c25278cf34a0392f256"
    )
    assert tuple(item["requirement_id"] for item in evidence["requirements"]) == (
        EXPECTED_REQUIREMENTS
    )
    assert len({item["requirement_id"] for item in evidence["requirements"]}) == len(
        EXPECTED_REQUIREMENTS
    )


def test_epic_4142_evidence_is_fail_closed_and_locally_traceable() -> None:
    evidence = _load()
    observed_counts = {status: 0 for status in ALLOWED_STATUSES}

    for item in evidence["requirements"]:
        status = item["status"]
        observed_counts[status] += 1
        assert status in ALLOWED_STATUSES
        assert item["requirement"].strip()
        assert item["rationale"].strip()
        assert item["validation_commands"]
        assert all(command.strip() for command in item["validation_commands"])
        assert item["evidence_files"]
        assert all("*" not in path for path in item["evidence_files"])
        for relative in item["evidence_files"]:
            assert (ROOT / relative).is_file(), relative

        if status == "verified":
            assert item["gaps"] == []
            assert any(
                path.startswith(("src/", "tests/"))
                and not path.startswith(NONAUTHORITATIVE_PREFIXES)
                for path in item["evidence_files"]
            )
            assert any(path.startswith("tests/") for path in item["evidence_files"])
        else:
            assert item["gaps"]
            assert all(gap.strip() for gap in item["gaps"])

    assert evidence["status_counts"] == observed_counts
    assert evidence["epic_closeable"] is False
    assert any(
        count for status, count in observed_counts.items() if status != "verified"
    )


def test_epic_4142_remote_evidence_is_immutable_and_reviewable() -> None:
    evidence = _load()

    for item in evidence["requirements"]:
        for remote in item["remote_evidence"]:
            assert remote.startswith("https://github.com/D-sorganization/")
            assert (
                "/actions/runs/" in remote or "/pull/" in remote or "/issues/" in remote
            )
            assert "/main/" not in remote
