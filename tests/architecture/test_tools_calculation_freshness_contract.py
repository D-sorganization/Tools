"""Consumer contracts for TOOLS-D6 calculation freshness and reverse impact."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from scripts.tools_calculation_freshness_contract import (
    CalculationFreshnessError,
    DriftError,
    ExemptionError,
    ReverseImpactError,
    TransitiveReverseImpactGate,
    load_calculation_freshness,
    verify_calculation_freshness,
    verify_expiring_exemptions,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FRESHNESS_PATH = REPO_ROOT / "manuals" / "tools" / "calculation-freshness.json"
SCHEMA_PATH = (
    REPO_ROOT / "manuals" / "tools" / "schemas" / "calculation-freshness.schema.json"
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_calculation_freshness_schema_is_strict_and_manifest_conforms() -> None:
    schema = _json(SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    manifest = _json(FRESHNESS_PATH)
    Draft202012Validator(schema).validate(manifest)

    assert schema["$id"].endswith("/tools/calculation-freshness/1.0.0.json")
    assert schema["additionalProperties"] is False


def test_freshness_loader_validates_structure_and_exemplar_coverage() -> None:
    manifest = load_calculation_freshness(_json(FRESHNESS_PATH))

    assert manifest.schema_version == "tools-calculation-freshness/1.0.0"
    assert manifest.manual_id == "tools"
    assert manifest.release_status == "provisional"
    assert manifest.owner_subepic == 4723
    assert len(manifest.calculations) == 1
    calc = manifest.calculations[0]
    assert calc.calculation_id == "TOOLS-DPLANE-GEOMETRY"
    assert calc.status == "verified-unapproved"

    categories = {ex.category for ex in calc.executable_examples}
    assert categories == {"nominal", "boundary", "failure"}


@pytest.mark.parametrize(
    ("mutation", "error_type", "message"),
    [
        (lambda v: v.update(extra=True), CalculationFreshnessError, "keys differ"),
        (
            lambda v: v.update(schema_version="invalid/1.0.0"),
            CalculationFreshnessError,
            "unsupported freshness schema_version",
        ),
        (
            lambda v: v.update(owner_subepic=9999),
            CalculationFreshnessError,
            "owner_subepic",
        ),
        (
            lambda v: v["calculations"][0].update(
                executable_examples=[
                    ex
                    for ex in v["calculations"][0]["executable_examples"]
                    if ex["category"] != "failure"
                ]
            ),
            CalculationFreshnessError,
            "failure examples",
        ),
        (
            lambda v: v["calculations"][0].update(
                executable_examples=[
                    ex
                    for ex in v["calculations"][0]["executable_examples"]
                    if ex["category"] != "boundary"
                ]
            ),
            CalculationFreshnessError,
            "boundary",
        ),
        (
            lambda v: v["calculations"][0].update(
                executable_examples=[
                    ex
                    for ex in v["calculations"][0]["executable_examples"]
                    if ex["category"] != "nominal"
                ]
            ),
            CalculationFreshnessError,
            "nominal",
        ),
    ],
)
def test_loader_rejects_malformed_manifest_or_missing_categories(
    mutation: Any, error_type: type[Exception], message: str
) -> None:
    data = copy.deepcopy(_json(FRESHNESS_PATH))
    mutation(data)
    with pytest.raises(error_type, match=message):
        load_calculation_freshness(data)


def test_expiring_exemptions_enforce_valid_unexpired_future_dates() -> None:
    manifest_data = copy.deepcopy(_json(FRESHNESS_PATH))
    now = datetime.now(UTC)

    # 1. Valid unexpired exemption
    manifest_data["exemptions"] = [
        {
            "exemption_id": "EXEMPT-20260907-DPLANE",
            "calculation_id": "TOOLS-DPLANE-GEOMETRY",
            "drift_type": "numerical-drift",
            "target_path": "manuals/tools/fixtures/dplane-calculation-fixtures.json",
            "rationale": "Temporary calibration adjustment pending upstream review.",
            "reviewed_by": ["@dieterolson"],
            "created_at": (now - timedelta(days=1)).isoformat(),
            "expires_at": (now + timedelta(days=5)).isoformat(),
            "status": "active",
        }
    ]
    manifest = load_calculation_freshness(manifest_data)
    active_count = verify_expiring_exemptions(manifest, current_time=now)
    assert active_count == 1

    # 2. Expired exemption fails closed
    expired_time = (now - timedelta(seconds=1)).isoformat()
    manifest_data["exemptions"][0]["expires_at"] = expired_time
    manifest_expired = load_calculation_freshness(manifest_data)
    with pytest.raises(ExemptionError, match="expired"):
        verify_expiring_exemptions(manifest_expired, current_time=now)

    # 3. Revoked exemption fails closed
    manifest_data["exemptions"][0]["expires_at"] = (now + timedelta(days=5)).isoformat()
    manifest_data["exemptions"][0]["status"] = "revoked"
    manifest_revoked = load_calculation_freshness(manifest_data)
    with pytest.raises(ExemptionError, match="revoked"):
        verify_expiring_exemptions(manifest_revoked, current_time=now)

    # 4. Placeholder rationale fails closed
    manifest_data["exemptions"][0]["status"] = "active"
    manifest_data["exemptions"][0]["rationale"] = "TODO: add explanation here later"
    with pytest.raises(ExemptionError, match="placeholder"):
        load_calculation_freshness(manifest_data)


def test_transitive_reverse_impact_graph_and_deleted_file_failure(
    tmp_path: Path,
) -> None:
    manifest = load_calculation_freshness(_json(FRESHNESS_PATH))
    gate = TransitiveReverseImpactGate(REPO_ROOT, manifest)

    # All tracked paths must exist in repository root
    gate.check_file_integrity()
    assert len(gate.tracked_paths) > 5

    # Check impacted calculation resolution
    impacted = gate.find_impacted_calculations(
        ["src/shared/python/swing_sim/impact/dplane.py"]
    )
    assert "TOOLS-DPLANE-GEOMETRY" in impacted

    # Moving or deleting a tracked file raises ReverseImpactError
    dummy_gate = TransitiveReverseImpactGate(tmp_path, manifest)
    with pytest.raises(
        ReverseImpactError, match="referenced file was moved or deleted"
    ):
        dummy_gate.check_file_integrity()


def test_full_calculation_freshness_verification_passes() -> None:
    summary = verify_calculation_freshness(REPO_ROOT)

    assert summary.calculation_count == 1
    assert summary.nominal_count >= 1
    assert summary.boundary_count >= 1
    assert summary.failure_count >= 1
    assert summary.documented_value_count >= 5
    assert summary.table_count >= 1
    assert summary.active_exemption_count == 0


def test_drift_without_exemption_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_data = copy.deepcopy(_json(FRESHNESS_PATH))
    # Mutate a documented value
    manifest_data["calculations"][0]["source_links"][0]["sha256_lf"] = "0" * 64
    mutated_file = REPO_ROOT / "manuals" / "tools" / "calculation-freshness.json"

    # Write temporarily and verify it fails with DriftError
    original_text = mutated_file.read_text(encoding="utf-8")
    try:
        mutated_file.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
        with pytest.raises(DriftError, match="source drift detected"):
            verify_calculation_freshness(REPO_ROOT)
    finally:
        mutated_file.write_text(original_text, encoding="utf-8")
