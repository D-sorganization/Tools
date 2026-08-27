"""Contract and parity tests against backend-authoritative golden fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from rate_of_closure.launch_monitor_v2_client import (
    validate_player_covariation_response,
    validate_strokes_gained_response,
    validate_v2_response,
)

FIXTURES_DIR = (
    Path(__file__).parents[2]
    / "src"
    / "rate_of_closure"
    / "web"
    / "src"
    / "model"
    / "__fixtures__"
)
CONFORMANCE_BUNDLE_PATH = (
    FIXTURES_DIR / "launch_monitor_conformance_bundle_golden_v1.json"
)
PLAYER_COVARIATION_PATH = (
    FIXTURES_DIR / "launch_monitor_player_covariation_golden_v1.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _walk_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*map(_walk_keys, value.values()))
    if isinstance(value, list):
        return set().union(*map(_walk_keys, value))
    return set()


def test_conformance_bundle_structure_and_no_embedded_rows() -> None:
    """Verify the bundle matches authoritative contract without raw shot rows."""
    bundle = _load_json(CONFORMANCE_BUNDLE_PATH)

    assert bundle["bundle_version"] == "launch-monitor-analytics-conformance/1.0.0"
    assert bundle["data_classification"] == (
        "synthetic_contract_fixture_no_private_rows"
    )
    assert bundle["input_records_embedded"] is False
    assert isinstance(bundle["bundle_sha256"], str)
    assert len(bundle["bundle_sha256"]) == 64

    scenarios = bundle["scenarios"]
    assert isinstance(scenarios, list)
    assert len(scenarios) == 10

    scenario_kinds = {
        (item["analysis_kind"], item["expected_status"]) for item in scenarios
    }
    expected_kinds = {
        ("analysis_v2", "available"),
        ("analysis_v2", "unavailable"),
        ("player_covariation", "available"),
        ("player_covariation", "unavailable"),
        ("attested_longitudinal", "available"),
        ("attested_longitudinal", "unavailable"),
        ("source_backed_strokes_gained", "available"),
        ("source_backed_strokes_gained", "unavailable"),
        ("distance_target_proxy", "available"),
        ("distance_target_proxy", "unavailable"),
    }
    assert scenario_kinds == expected_kinds

    all_keys = _walk_keys(bundle)
    assert "records" not in all_keys
    assert "restricted_internal" not in json.dumps(bundle)


def test_conformance_bundle_scenarios_retain_units_claims_and_lineage() -> None:
    """Verify each scenario preserves unit contracts, claims, and backing lineage."""
    bundle = _load_json(CONFORMANCE_BUNDLE_PATH)

    for scenario in bundle["scenarios"]:
        assert scenario["units"]
        claims = scenario["claims"]
        assert claims["causal_inference"] is False
        if "device_emulation" in claims:
            assert claims["device_emulation"] is False
        if "device_certification" in claims:
            assert claims["device_certification"] is False

        sources = scenario["sources"]
        backing_records = scenario["backing_records"]
        assert isinstance(sources, list) and sources
        assert isinstance(backing_records, list) and backing_records

        source_ids = {s["source_id"] for s in sources}
        assert all(r["source_id"] in source_ids for r in backing_records)
        assert all(len(r["record_sha256"]) == 64 for r in backing_records)
        assert sum(scenario["exclusions"].values()) >= 0


def test_conformance_bundle_validates_against_v2_client_validators() -> None:
    """Validate each supported scenario through the rate_of_closure v2 validators."""
    bundle = _load_json(CONFORMANCE_BUNDLE_PATH)
    by_kind = {s["scenario_id"]: s for s in bundle["scenarios"]}

    analysis_avail = by_kind["analysis-v2-available"]
    validated_analysis = validate_v2_response(analysis_avail["payload"])
    assert validated_analysis.contract_version == "2.0.0"
    assert validated_analysis.payload["status"] == "available"

    cov_avail = by_kind["player-covariation-available"]
    validated_cov = validate_player_covariation_response(cov_avail["payload"])
    assert (
        validated_cov["contract_version"] == "launch-monitor-player-covariation/1.0.0"
    )
    assert validated_cov["status"] == "available"

    sg_avail = by_kind["source-backed-strokes-gained-available"]
    validated_sg = validate_strokes_gained_response(sg_avail["payload"])
    assert (
        validated_sg.payload["contract_version"]
        == "launch-monitor-strokes-gained-analysis/1.0.0"
    )
    assert validated_sg.status == "available"
    assert validated_sg.count == 3
    assert validated_sg.payload["claims"]["source_backed"] is True
    assert validated_sg.payload["claims"]["is_strokes_gained"] is True

    proxy_avail = by_kind["distance-target-proxy-available"]
    assert proxy_avail["claims"]["is_strokes_gained"] is False
    assert proxy_avail["claims"]["source_backed"] is False


def test_player_covariation_golden_fixture_parity() -> None:
    """Verify the player covariation golden fixture against canonical validator."""
    fixture = _load_json(PLAYER_COVARIATION_PATH)

    assert (
        fixture["fixture_version"] == "launch-monitor-player-covariation-golden/1.0.0"
    )
    assert "Synthetic aggregation-reversal" in fixture["description"]

    records = fixture["records"]
    assert isinstance(records, list)
    assert len(records) == 10

    context = fixture["context"]
    assert context["player_identity"]["trust_level"] == "explicit_user_attested"
    assert context["player_identity"]["identifier_column"] == "player_id"

    expected_result = fixture["expected_result"]
    validated = validate_player_covariation_response(expected_result)
    assert validated["contract_version"] == "launch-monitor-player-covariation/1.0.0"
    assert validated["status"] == "available"
    assert validated["claims"]["causal_inference"] is False
    assert validated["claims"]["device_emulation"] is False

    expected_scan = fixture["expected_scan_result"]
    validated_scan = validate_player_covariation_response(expected_scan)
    assert validated_scan["analysis_kind"] == "pair_scan"
    assert validated_scan["status"] == "available"
