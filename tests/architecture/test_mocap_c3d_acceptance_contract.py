"""Acceptance-only contract for dependency-gated C3D exchange work."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.contract]

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_CONTRACT_PATH = (
    _REPOSITORY_ROOT / "docs" / "development" / "mocap_c3d_exchange_acceptance.json"
)


def _contract() -> dict[str, object]:
    payload = json.loads(_CONTRACT_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return cast(dict[str, object], payload)


def _mapping(parent: dict[str, object], key: str) -> dict[str, object]:
    value = parent[key]
    assert isinstance(value, dict), f"{key} must be an object"
    return cast(dict[str, object], value)


def _records(parent: dict[str, object], key: str) -> list[dict[str, object]]:
    value = parent[key]
    assert isinstance(value, list), f"{key} must be an array"
    assert all(isinstance(item, dict) for item in value)
    return cast(list[dict[str, object]], value)


def _strings(parent: dict[str, object], key: str) -> set[str]:
    value = parent[key]
    assert isinstance(value, list), f"{key} must be an array"
    assert all(isinstance(item, str) for item in value)
    return set(cast(list[str], value))


def test_contract_is_acceptance_only_and_dependency_gated() -> None:
    """The artifact must not become a competing runtime or schema authority."""
    contract = _contract()
    authority = _mapping(contract, "authority")
    readiness = _mapping(contract, "readiness")

    assert contract["schema_version"] == "tools-m9-acceptance/1.0.0"
    assert contract["artifact_role"] == "acceptance-only"
    assert authority["runtime_owner"] == "D-sorganization/Tools"
    assert authority["canonical_input"] == "sidekick.lab.mocap.MocapSession"
    assert authority["public_runtime_schema"] is False
    assert readiness["implementation_eligible"] is False
    assert readiness["local_stack_policy"] == "evidence-only-never-merge"

    dependencies = _records(readiness, "required_dependencies")
    assert {record["issue"] for record in dependencies} == {4708, 4710}
    assert all(record["state"] == "open" for record in dependencies)
    assert all(record["gate"] == "merged-to-protected-main" for record in dependencies)


def test_contract_requires_complete_normalized_semantics() -> None:
    """C3D conformance compares normalized meaning rather than file bytes."""
    semantics = _mapping(_contract(), "semantic_profile")
    required_families = {
        "frames",
        "units",
        "timestamps",
        "confidence",
        "skeleton",
        "events",
        "residuals",
        "analog",
        "force",
        "provenance",
        "unknown_metadata",
    }

    assert _strings(semantics, "required_families") == required_families
    assert semantics["comparison"] == "normalized-semantics"
    assert semantics["byte_equality_required"] is False
    assert semantics["unknown_metadata"] == "preserve-or-reject-explicitly"
    assert semantics["deterministic_round_trip"] == (
        "write-read-write-normalized-stable"
    )


def test_contract_makes_camera_mask_overflow_unrepresentable() -> None:
    """Eight or more contributors must never become an invented seven-camera subset."""
    policy = _mapping(_contract(), "camera_contributor_policy")
    cases = _records(policy, "cases")
    by_count = {str(record["contributors"]): record for record in cases}

    assert policy["standard_capacity"] == 7
    assert policy["overflow_policy"] == "never-truncate-wrap-remap-or-invent"
    for count in ("0", "1", "7"):
        assert by_count[count]["standard_mask"] == "exact"
        assert by_count[count]["sidecar"] == "optional"
    for count in ("8", "N>8"):
        assert by_count[count]["standard_mask"] == "unavailable-zero"
        assert by_count[count]["sidecar"] == "required"


def test_contract_requires_digest_bound_external_loss_sidecar() -> None:
    """Unrepresentable evidence remains canonical and tamper evident."""
    sidecar = _mapping(_contract(), "loss_sidecar")

    assert sidecar["format"] == "canonical-json"
    assert sidecar["location"] == "adjacent-external"
    assert sidecar["missing_or_tampered"] == "fail-closed"
    assert sidecar["round_trip"] == "write-read-write-stable"
    assert _strings(sidecar, "required_bindings") == {
        "c3d_sha256",
        "canonical_session_sha256",
        "semantic_profile_sha256",
    }
    assert _strings(sidecar, "required_payloads") == {
        "complete_contributor_ids",
        "losses",
        "writer_provenance",
    }


def test_contract_requires_nonoptional_independent_oracles() -> None:
    """Missing interoperability tooling must fail qualification, not skip it."""
    oracles = _records(_contract(), "oracles")
    by_id = {str(record["id"]): record for record in oracles}

    assert set(by_id) == {"ezc3d", "python-c3d", "btk", "c3d.org-sample-corpus"}
    assert all(record["required"] is True for record in oracles)
    assert by_id["python-c3d"]["role"] == "independent-reader"
    assert by_id["btk"]["role"] == "legacy-compatibility"
    assert by_id["c3d.org-sample-corpus"]["role"] == "normative-corpus"


def test_contract_rejects_every_audited_stale_behavior() -> None:
    """The local M9 prototype is evidence only until each conflict is removed."""
    assert _strings(_contract(), "forbidden_behaviors") == {
        "competing-c3d-record-schema",
        "byte-equality-as-conformance",
        "optional-or-skipped-independent-oracle",
        "first-seven-overflow-truncation",
        "embedded-extension-as-only-loss-record",
        "missing-normalized-semantic-comparison",
        "missing-reference-corpus",
    }
