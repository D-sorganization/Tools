"""Strict contract tests for the four-surface capability inventory."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from rate_of_closure.four_surface_capability import (
    CAPABILITY_CATEGORIES,
    SURFACE_IDS,
    FourSurfaceCapabilityManifest,
    canonical_manifest_json,
    load_four_surface_capability,
    render_json_schema,
    validate_freshness,
    validate_repository_evidence,
)
from rate_of_closure.four_surface_declarations import (
    derive_declared_capabilities,
    render_declared_scope,
    validate_declared_scope_completeness,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "docs" / "release" / "four_surface_capability.v1.json"
SCHEMA_PATH = REPO_ROOT / "docs" / "release" / "four_surface_capability.v1.schema.json"
CAMPAIGN_PATH = REPO_ROOT / "docs" / "release" / "rate_of_closure_campaign.v1.json"


@pytest.fixture
def manifest() -> FourSurfaceCapabilityManifest:
    """Load the checked-in partial capability inventory."""
    return load_four_surface_capability(MANIFEST_PATH)


@pytest.mark.unit
def test_surface_ids_and_declared_categories_are_complete(
    manifest: FourSurfaceCapabilityManifest,
) -> None:
    """Every declared capability must classify all stable product surfaces."""
    assert tuple(surface.id.value for surface in manifest.surfaces) == SURFACE_IDS
    assert {capability.category.value for capability in manifest.capabilities} == set(
        CAPABILITY_CATEGORIES
    )
    for capability in manifest.capabilities:
        assert set(capability.surfaces) == set(SURFACE_IDS)


@pytest.mark.unit
def test_unsupported_cells_have_reasons_and_supported_cells_have_evidence(
    manifest: FourSurfaceCapabilityManifest,
) -> None:
    """The matrix may not imply support or hide unsupported behavior."""
    for capability in manifest.capabilities:
        for surface in capability.surfaces.values():
            if surface.state.value == "supported":
                assert surface.reason is None
                assert surface.evidence_ids
            else:
                assert surface.reason


@pytest.mark.unit
def test_supported_without_evidence_is_rejected(
    manifest: FourSurfaceCapabilityManifest,
) -> None:
    """A supported claim requires commit-bound evidence."""
    payload = manifest.model_dump(mode="json")
    payload["capabilities"][0]["surfaces"]["tools.pyqt6"] = {
        "state": "supported",
        "reason": None,
        "evidence_ids": [],
    }
    with pytest.raises(ValidationError, match="supported state requires evidence"):
        FourSurfaceCapabilityManifest.model_validate_json(json.dumps(payload))


@pytest.mark.unit
def test_unsupported_without_reason_is_rejected(
    manifest: FourSurfaceCapabilityManifest,
) -> None:
    """Unsupported behavior must remain visible to downstream consumers."""
    payload = manifest.model_dump(mode="json")
    payload["capabilities"][0]["surfaces"]["upstreamdrift.pyqt6"]["reason"] = None
    with pytest.raises(ValidationError, match="unsupported state requires a reason"):
        FourSurfaceCapabilityManifest.model_validate_json(json.dumps(payload))


@pytest.mark.unit
def test_duplicate_capability_ids_are_rejected(
    manifest: FourSurfaceCapabilityManifest,
) -> None:
    """Capability IDs are durable keys and therefore globally unique."""
    payload = manifest.model_dump(mode="json")
    payload["capabilities"].append(copy.deepcopy(payload["capabilities"][0]))
    with pytest.raises(ValidationError, match="duplicate capability id"):
        FourSurfaceCapabilityManifest.model_validate_json(json.dumps(payload))


@pytest.mark.unit
def test_json_primitive_types_are_strict(
    manifest: FourSurfaceCapabilityManifest,
) -> None:
    """Numeric text cannot silently satisfy a numeric freshness field."""
    payload = manifest.model_dump(mode="json", by_alias=True)
    payload["freshness"]["max_age_days"] = "30"
    with pytest.raises(ValidationError, match="valid integer"):
        FourSurfaceCapabilityManifest.model_validate_json(json.dumps(payload))


@pytest.mark.unit
def test_evidence_must_match_exact_tools_pin(
    manifest: FourSurfaceCapabilityManifest,
) -> None:
    """Evidence from a different Tools tree cannot support this snapshot."""
    payload = manifest.model_dump(mode="json", by_alias=True)
    payload["evidence"][0]["commit_sha"] = "a" * 40
    with pytest.raises(ValidationError, match="exact Tools pin"):
        FourSurfaceCapabilityManifest.model_validate_json(json.dumps(payload))


@pytest.mark.unit
def test_consumer_support_without_installed_pin_is_rejected(
    manifest: FourSurfaceCapabilityManifest,
) -> None:
    """A launcher or copied route cannot become consumer support by assertion."""
    payload = manifest.model_dump(mode="json", by_alias=True)
    payload["capabilities"][0]["surfaces"]["upstreamdrift.pyqt6"] = {
        "state": "supported",
        "reason": None,
        "evidence_ids": ["tools-pyqt-capability-workflow"],
    }
    with pytest.raises(ValidationError, match="installed consumer pin"):
        FourSurfaceCapabilityManifest.model_validate_json(json.dumps(payload))


@pytest.mark.unit
def test_consumer_support_requires_evidence_from_consumer_pin(
    manifest: FourSurfaceCapabilityManifest,
) -> None:
    """A consumer pin cannot reuse Tools-only evidence as conformance proof."""
    payload = manifest.model_dump(mode="json", by_alias=True)
    payload["surfaces"][2]["consumer_commit_sha"] = "b" * 40
    payload["capabilities"][0]["surfaces"]["upstreamdrift.pyqt6"] = {
        "state": "supported",
        "reason": None,
        "evidence_ids": ["tools-pyqt-capability-workflow"],
    }
    with pytest.raises(ValidationError, match="evidence from the surface pin"):
        FourSurfaceCapabilityManifest.model_validate_json(json.dumps(payload))


@pytest.mark.unit
def test_missing_surface_and_category_are_rejected(
    manifest: FourSurfaceCapabilityManifest,
) -> None:
    """Completeness gates reject both a missing cell and a missing category."""
    payload = manifest.model_dump(mode="json", by_alias=True)
    del payload["capabilities"][0]["surfaces"]["tools.react"]
    with pytest.raises(ValidationError, match="all canonical IDs"):
        FourSurfaceCapabilityManifest.model_validate_json(json.dumps(payload))

    payload = manifest.model_dump(mode="json", by_alias=True)
    payload["capabilities"] = [
        item for item in payload["capabilities"] if item["category"] != "export"
    ]
    with pytest.raises(ValidationError, match="every declared category"):
        FourSurfaceCapabilityManifest.model_validate_json(json.dumps(payload))


@pytest.mark.unit
def test_evidence_is_exactly_pinned_and_resolves_in_tools(
    manifest: FourSurfaceCapabilityManifest,
) -> None:
    """Supported Tools claims resolve only against the exact audited commit."""
    validate_repository_evidence(manifest, REPO_ROOT)
    assert all(
        item.commit_sha == manifest.tools_pin.commit_sha
        for item in manifest.evidence
        if item.repository == manifest.tools_pin.repository
    )


@pytest.mark.unit
def test_schema_and_canonical_output_are_deterministic(
    manifest: FourSurfaceCapabilityManifest,
) -> None:
    """Generated artifacts must be byte-stable for CI and consumers."""
    schema_bytes = render_json_schema()
    assert SCHEMA_PATH.read_bytes() == schema_bytes
    assert manifest.schema_ref.sha256 == hashlib.sha256(schema_bytes).hexdigest()
    assert canonical_manifest_json(manifest) == canonical_manifest_json(
        load_four_surface_capability(MANIFEST_PATH)
    )


@pytest.mark.unit
def test_schema_is_versioned_and_strict() -> None:
    """The public schema rejects unknown top-level fields."""
    schema = json.loads(render_json_schema())
    assert schema["properties"]["schema_version"]["const"] == (
        "four-surface-capability/v1"
    )
    assert schema["additionalProperties"] is False


@pytest.mark.unit
def test_freshness_boundary_fails_closed(
    manifest: FourSurfaceCapabilityManifest,
) -> None:
    """CI can reject observations before their date or after their expiry."""
    validate_freshness(manifest, on_date=manifest.freshness.observed_on)
    validate_freshness(manifest, on_date=manifest.freshness.expires_on)
    with pytest.raises(ValueError, match="not yet current"):
        validate_freshness(manifest, on_date=date(2026, 8, 8))
    with pytest.raises(ValueError, match="stale"):
        validate_freshness(manifest, on_date=date(2026, 9, 9))


@pytest.mark.unit
def test_campaign_manifest_links_partial_inventory_without_release_claim() -> None:
    """The canonical campaign authority references but does not overstate the slice."""
    campaign = json.loads(CAMPAIGN_PATH.read_text(encoding="utf-8"))
    program = next(item for item in campaign["programs"] if item["issue"] == 4260)
    authority_paths = {
        item["value"]
        for item in program["authorities"]
        if item["kind"] == "repository_path"
    }
    assert "docs/release/four_surface_capability.v1.json" in authority_paths
    assert program["completion"] == "specified"
    assert program["delivery_stage"] == "specified_only"
    assert program["release"]["status"] == "not_released"


@pytest.mark.unit
def test_declared_scope_is_complete_and_deterministic(
    manifest: FourSurfaceCapabilityManifest,
) -> None:
    """All structured campaign programs and linked active specs are enumerated."""
    declared = derive_declared_capabilities(CAMPAIGN_PATH, REPO_ROOT)
    validate_declared_scope_completeness(manifest, CAMPAIGN_PATH, REPO_ROOT)
    # 15 campaign programs + 24 linked active specs. The ground families added
    # five linked specifications (material profiles, impact bounce, skid roll,
    # reference execution, result studies).
    assert len(declared) == 39
    assert manifest.inventory.status == "declared_scope_complete"
    assert manifest.inventory.campaign_program_count == 15
    assert manifest.inventory.active_specification_count == 24
    assert manifest.inventory.curated_capability_count == 6
    assert render_declared_scope(CAMPAIGN_PATH, REPO_ROOT) == render_declared_scope(
        CAMPAIGN_PATH, REPO_ROOT
    )


@pytest.mark.unit
def test_every_declared_record_has_four_truthful_cells(
    manifest: FourSurfaceCapabilityManifest,
) -> None:
    """Generated-scope records cannot omit or overstate a product surface."""
    declared_ids = {
        item.id for item in derive_declared_capabilities(CAMPAIGN_PATH, REPO_ROOT)
    }
    records = {item.id: item for item in manifest.capabilities}
    assert declared_ids <= records.keys()
    for capability_id in declared_ids:
        record = records[capability_id]
        assert set(record.surfaces) == set(SURFACE_IDS)
        for cell in record.surfaces.values():
            if cell.state.value == "supported":
                assert cell.evidence_ids
            else:
                assert cell.reason


@pytest.mark.unit
def test_declared_unsupported_reasons_identify_their_source(
    manifest: FourSurfaceCapabilityManifest,
) -> None:
    """Unsupported reasons are record-specific rather than generic boilerplate."""
    for record in manifest.capabilities:
        if record.category.value == "campaign_program":
            marker = f"#{record.id.removeprefix('campaign.issue-')}"
        elif record.category.value == "active_specification":
            marker = record.declaration.source_path
        else:
            continue
        for cell in record.surfaces.values():
            if cell.state.value == "unsupported":
                assert cell.reason is not None
                assert marker in cell.reason


@pytest.mark.unit
def test_declared_record_kind_must_match_derived_source(
    manifest: FourSurfaceCapabilityManifest,
) -> None:
    """A valid but incorrect declaration kind cannot satisfy completeness."""
    records = list(manifest.capabilities)
    index = next(
        index
        for index, item in enumerate(records)
        if item.category.value == "campaign_program"
    )
    campaign = records[index]
    declaration = campaign.declaration.model_copy(
        update={"kind": "active_release_spec"}
    )
    records[index] = campaign.model_copy(update={"declaration": declaration})
    changed = manifest.model_copy(update={"capabilities": records})
    with pytest.raises(ValueError, match="declared capability metadata differs"):
        validate_declared_scope_completeness(changed, CAMPAIGN_PATH, REPO_ROOT)


@pytest.mark.unit
def test_new_campaign_program_cannot_bypass_completeness_gate(
    manifest: FourSurfaceCapabilityManifest,
    tmp_path: Path,
) -> None:
    """Adding a structured campaign record requires a matrix record."""
    payload = json.loads(CAMPAIGN_PATH.read_text(encoding="utf-8"))
    added = copy.deepcopy(payload["programs"][0])
    added["issue"] = 4999
    added["title"] = "New Structured Rate Capability"
    payload["programs"].append(added)
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="declared capability IDs differ"):
        validate_declared_scope_completeness(manifest, campaign_path, REPO_ROOT)


@pytest.mark.unit
def test_new_linked_spec_cannot_bypass_completeness_gate(
    manifest: FourSurfaceCapabilityManifest,
    tmp_path: Path,
) -> None:
    """Adding a linked active specification requires a matrix record."""
    payload = json.loads(CAMPAIGN_PATH.read_text(encoding="utf-8"))
    spec_root = tmp_path / "docs" / "specs"
    spec_root.mkdir(parents=True)
    repository_spec_root = REPO_ROOT / "docs" / "specs"
    for source in repository_spec_root.rglob("*.md"):
        destination = spec_root / source.relative_to(repository_spec_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    shutil.copyfile(REPO_ROOT / "SPEC.md", tmp_path / "SPEC.md")
    rate_docs = tmp_path / "docs" / "rate_of_closure"
    rate_docs.mkdir(parents=True)
    shutil.copyfile(
        REPO_ROOT
        / "docs"
        / "rate_of_closure"
        / "variation_visualization_performance.md",
        rate_docs / "variation_visualization_performance.md",
    )
    new_spec = spec_root / "NEW_RATE_FEATURE.md"
    new_spec.write_text("# New Rate Feature\n", encoding="utf-8")
    payload["programs"][0]["authorities"].append(
        {
            "kind": "repository_path",
            "value": "docs/specs/NEW_RATE_FEATURE.md",
        }
    )
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="declared capability IDs differ"):
        validate_declared_scope_completeness(manifest, campaign_path, tmp_path)


@pytest.mark.unit
def test_linked_spec_path_cannot_escape_declared_scope(
    tmp_path: Path,
) -> None:
    """A structured spec authority cannot traverse outside its governed root."""
    payload = json.loads(CAMPAIGN_PATH.read_text(encoding="utf-8"))
    payload["programs"][0]["authorities"].append(
        {
            "kind": "repository_path",
            "value": "docs/specs/../ESCAPE.md",
        }
    )
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="path must be normalized"):
        derive_declared_capabilities(campaign_path, REPO_ROOT)
