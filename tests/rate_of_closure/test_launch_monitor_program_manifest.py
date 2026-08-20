"""Contract tests for the cross-repository launch-monitor release program."""

from __future__ import annotations

import json
import re
from pathlib import Path

MANIFEST_PATH = (
    Path(__file__).parents[2] / "docs" / "release" / "launch_monitor_program.v1.json"
)
REQUIRED_REPOSITORIES = {
    "D-sorganization/Tools",
    "D-sorganization/UpstreamDrift",
    "D-sorganization/AffineDrift",
    "D-sorganization/Launch-Monitor-Data",
    "D-sorganization/Launch-Monitor-Flight-Model-Campaign",
}
REQUIRED_POLICY_IDS = {
    "canonical-statistics-authority",
    "shotlink-training-quarantine",
    "release-a-release-b-boundary",
    "explicit-player-identity",
}
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _manifest() -> dict[str, object]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_program_manifest_pins_authorities_and_approved_policies() -> None:
    manifest = _manifest()

    assert manifest["schema_version"] == 1
    assert manifest["program_issue"] == "D-sorganization/Tools#4583"
    repositories = manifest["repositories"]
    assert isinstance(repositories, list)
    assert {item["name"] for item in repositories} == REQUIRED_REPOSITORIES
    assert all(SHA_PATTERN.fullmatch(item["main_commit"]) for item in repositories)
    policies = manifest["policies"]
    assert isinstance(policies, list)
    assert {item["id"] for item in policies} == REQUIRED_POLICY_IDS
    assert all(item["status"] == "approved" for item in policies)


def test_program_manifest_has_separate_software_and_validation_releases() -> None:
    releases = _manifest()["releases"]

    assert isinstance(releases, list)
    by_id = {item["id"]: item for item in releases}
    assert set(by_id) == {"release-a", "release-b"}
    assert by_id["release-a"]["requires_paired_device_data"] is False
    assert by_id["release-b"]["requires_paired_device_data"] is True
    assert by_id["release-a"]["completion_evidence"]
    assert by_id["release-b"]["completion_evidence"]


def test_program_manifest_assigns_one_owner_to_each_capability() -> None:
    capabilities = _manifest()["capabilities"]

    assert isinstance(capabilities, list)
    assert len({item["id"] for item in capabilities}) == len(capabilities)
    assert all(item["owner"] in REQUIRED_REPOSITORIES for item in capabilities)
    assert all(item["tracking_issue"] for item in capabilities)
    assert all(
        item["status"] in {"planned", "in_progress", "complete"}
        for item in capabilities
    )
