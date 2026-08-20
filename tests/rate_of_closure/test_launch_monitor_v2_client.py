from __future__ import annotations

import pytest

from rate_of_closure.launch_monitor_v2_client import validate_v2_response


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
