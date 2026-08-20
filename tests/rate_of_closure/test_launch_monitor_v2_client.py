from __future__ import annotations

import pytest

from rate_of_closure.launch_monitor_v2_client import (
    UpstreamV2Client,
    validate_strokes_gained_response,
    validate_v2_response,
)


def _sg_response() -> dict[str, object]:
    return {
        "contract_version": "launch-monitor-strokes-gained-analysis/1.0.0",
        "status": "available",
        "metric_name": "source_backed_strokes_gained",
        "unit": "strokes",
        "value_summary": {"count": 3, "mean": 0.25},
        "baseline": {
            "baseline_id": "test",
            "version": "1",
            "source_url": "https://example.org",
            "license": "test",
            "table_sha256": "a" * 64,
            "contract_version": "launch-monitor-strokes-gained-baseline/2.0.0",
        },
        "formula": "SG = E(start) - 1 - E(finish)",
        "units": {"strokes_gained": "strokes", "distance": "yd"},
        "availability": {
            "state": "available",
            "observed_count": 3,
            "required_count": 3,
        },
        "uncertainty": {
            "sampling_method": "student-t",
            "confidence_level": 0.95,
            "benchmark_method": "unavailable",
            "assumptions": [],
        },
        "row_results": [],
        "excluded_rows": [],
        "exclusions": {
            "input_row_count": 3,
            "included_row_count": 3,
            "total_excluded": 0,
            "by_reason": {},
        },
        "group_summaries": [],
        "longitudinal_summaries": [],
        "analysis_context": {
            "player_identity": {"trust_level": "not_provided"},
            "sources": [],
            "transforms": [],
            "source_units": {},
        },
        "dataset_fingerprint_sha256": "b" * 64,
        "claims": {
            "is_strokes_gained": True,
            "source_backed": True,
            "device_emulation": False,
            "device_certification": False,
            "causal_inference": False,
        },
        "warnings": [],
        "limitations": [],
    }


def _response() -> dict[str, object]:
    return {
        "contract_version": "2.0.0",
        "status": "available",
        "analysis": {},
        "units": {},
        "lineage": {"dataset_fingerprint_sha256": "a" * 64, "backing_records": []},
        "missingness": {
            "input_row_count": 0,
            "complete_row_count": 0,
            "missing_by_variable": {},
            "non_numeric_by_variable": {},
            "excluded_by_reason": {},
            "policy": "pairwise",
        },
        "availability": [],
        "uncertainty": {
            "confidence_level": 0.95,
            "correlation_interval": "fisher-z",
            "regression_interval": "student-t",
            "multiplicity_adjustment": "benjamini-hochberg",
            "assumptions": [],
        },
        "player_identity": {"trust_level": "not_provided"},
        "vendor_provenance": [],
        "claims": {
            "vendor_comparison": "descriptive",
            "device_emulation": False,
            "device_certification": False,
            "causal_inference": False,
        },
        "warnings": [],
    }


def test_v2_client_validates_contract_and_typed_residual_availability() -> None:
    result = validate_v2_response(_response())
    assert result.contract_version == "2.0.0"
    assert result.row_aligned_residuals.state == "unavailable"
    assert "row-aligned" in result.row_aligned_residuals.reason


def test_v2_client_rejects_unknown_or_claim_unsafe_response() -> None:
    response = _response()
    response["contract_version"] = "1.0.0"
    with pytest.raises(ValueError, match="contract"):
        validate_v2_response(response)
    response = _response()
    response["claims"] = {**response["claims"], "device_emulation": True}  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="emulation"):
        validate_v2_response(response)


def test_v2_client_rejects_non_http_authorities_and_invalid_timeouts() -> None:
    with pytest.raises(ValueError, match=r"HTTP\(S\)"):
        UpstreamV2Client("file:///tmp/private.json")
    with pytest.raises(ValueError, match="positive"):
        UpstreamV2Client("https://analysis.example", timeout_seconds=0)


def test_strokes_gained_client_validates_scoring_contract_and_claims() -> None:
    result = validate_strokes_gained_response(_sg_response())
    assert result.mean == pytest.approx(0.25)
    assert result.count == 3
    assert result.payload["baseline"]["baseline_id"] == "test"  # type: ignore[index]
    unsafe = _sg_response()
    unsafe["claims"] = {**unsafe["claims"], "source_backed": False}  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="source-backed"):
        validate_strokes_gained_response(unsafe)
