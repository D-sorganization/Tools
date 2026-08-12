"""Contract tests for the Rate of Closure campaign release manifest."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.rate_campaign_manifest import (
    CampaignManifest,
    load_campaign_manifest,
    validate_repository_evidence,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "docs" / "release" / "rate_of_closure_campaign.v1.json"
PRIMARY_ISSUES = {
    4103,
    4120,
    4125,
    4130,
    4142,
    4146,
    4158,
    4181,
    4189,
    4191,
    4201,
    4218,
    4234,
    4260,
    4267,
}


@pytest.fixture
def manifest() -> CampaignManifest:
    """Load the checked-in campaign authority."""
    return load_campaign_manifest(MANIFEST_PATH)


@pytest.mark.unit
def test_manifest_covers_primary_campaign_issues(
    manifest: CampaignManifest,
) -> None:
    """Every primary campaign program must have one authoritative record."""
    assert {program.issue for program in manifest.programs} == PRIMARY_ISSUES


@pytest.mark.unit
def test_manifest_distinguishes_delivery_stages(
    manifest: CampaignManifest,
) -> None:
    """A feature-stack implementation must not masquerade as a release."""
    stages = {stage.value for stage in manifest.release_stage_definitions}
    assert stages == {
        "specified_only",
        "implemented_on_feature_stack",
        "protected_merged_to_parent",
        "released_to_main",
    }
    assert all(program.release.sha is None for program in manifest.programs)
    assert all(
        program.release.status.value == "not_released" for program in manifest.programs
    )


@pytest.mark.unit
def test_released_state_requires_main_sha(manifest: CampaignManifest) -> None:
    """Contradictory release claims must fail closed."""
    payload = manifest.model_dump(mode="json")
    payload["programs"][0]["delivery_stage"] = "released_to_main"
    payload["programs"][0]["release"] = {
        "status": "released",
        "branch": "main",
        "sha": None,
    }
    with pytest.raises(ValidationError, match="40-character release SHA"):
        CampaignManifest.model_validate(payload)


@pytest.mark.unit
def test_campaign_release_requires_every_program_release(
    manifest: CampaignManifest,
) -> None:
    """An umbrella release cannot contradict its component programs."""
    payload = manifest.model_dump(mode="json")
    payload["campaign_release"] = {
        "status": "released",
        "branch": "main",
        "sha": "a" * 40,
    }
    with pytest.raises(ValidationError, match="campaign release"):
        CampaignManifest.model_validate(payload)


@pytest.mark.unit
def test_placeholders_are_rejected(manifest: CampaignManifest) -> None:
    """Narrative placeholders cannot enter release evidence."""
    payload = copy.deepcopy(manifest.model_dump(mode="json"))
    payload["programs"][0]["limitations"] = ["TBD after the next run"]
    with pytest.raises(ValidationError, match="placeholder"):
        CampaignManifest.model_validate(payload)


@pytest.mark.unit
def test_dependency_cycles_are_rejected(manifest: CampaignManifest) -> None:
    """The declared implementation order must remain acyclic."""
    payload = manifest.model_dump(mode="json")
    payload["programs"][0]["depends_on_issues"] = [4120]
    with pytest.raises(ValidationError, match="dependency cycle"):
        CampaignManifest.model_validate(payload)


@pytest.mark.unit
def test_repository_evidence_references_exist(
    manifest: CampaignManifest,
) -> None:
    """Specs, carrier IDs, and evidence IDs must resolve locally."""
    validate_repository_evidence(manifest, REPO_ROOT)


@pytest.mark.unit
def test_schema_is_versioned_and_forbids_extra_fields() -> None:
    """The generated consumer schema must be strict and versioned."""
    schema = CampaignManifest.model_json_schema()
    assert schema["properties"]["schema_version"]["const"] == (
        "rate-of-closure-campaign/v1"
    )
    assert schema["additionalProperties"] is False
