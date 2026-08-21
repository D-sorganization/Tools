from __future__ import annotations

import json
from pathlib import Path

import pytest

from rate_of_closure.launch_monitor_v2_client import (
    MAX_CANONICAL_INLINE_RECORDS,
    UpstreamV2Client,
    build_dataset_job_request,
    build_player_covariation_payload,
    load_canonical_dataset_reference,
    validate_dataset_job_page,
    validate_dataset_job_status,
    validate_player_covariation_response,
    validate_strokes_gained_response,
    validate_v2_response,
)

GOLDEN = (
    Path(__file__).resolve().parents[2]
    / "src/rate_of_closure/launch_monitor_canonical_v2_golden.json"
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


def test_canonical_dataset_reference_and_job_contract_match_shared_golden() -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    reference = load_canonical_dataset_reference(golden["dataset_reference"])

    assert reference.expected_row_count == 261_666
    assert (
        build_dataset_job_request(reference, "source_summary")
        == golden["dataset_job_request"]
    )
    with pytest.raises(ValueError, match="root_id"):
        load_canonical_dataset_reference(
            {**golden["dataset_reference"], "root_id": "../private/path"}
        )
    with pytest.raises(ValueError, match="canonical dataset metrics"):
        build_dataset_job_request(reference, "metric_summary", metrics=("bogus",))


def test_dataset_job_responses_are_versioned_bounded_and_data_free() -> None:
    status = validate_dataset_job_status(
        {
            "contract_version": "launch-monitor-dataset-job/1.0.0",
            "job_id": "a" * 32,
            "status": "completed",
            "submitted_at_utc": "2026-08-21T00:00:00Z",
            "completed_at_utc": "2026-08-21T00:00:01Z",
            "input_row_count": 261_666,
            "result_item_count": 1,
            "unavailable": None,
        }
    )
    assert status["input_row_count"] == 261_666
    page = validate_dataset_job_page(
        {
            "contract_version": "launch-monitor-dataset-job/1.0.0",
            "job_id": "a" * 32,
            "offset": 0,
            "limit": 100,
            "total_items": 1,
            "next_offset": None,
            "items": [
                {
                    "source_id": "trackman",
                    "row_count": 11_699,
                    "vendor_key": "trackman",
                    "redistribution_status": "restricted",
                    "license_spdx": "NOASSERTION",
                    "backing_repository": "owner/repository",
                    "backing_commit": "b" * 40,
                    "backing_object_digests": [],
                }
            ],
        }
    )
    assert page["items"][0]["row_count"] == 11_699
    with pytest.raises(ValueError, match="private rows"):
        validate_dataset_job_page(
            {**page, "items": [{"shot_id": "secret", "ball_speed": 170.0}]}
        )
    with pytest.raises(ValueError, match="aggregate schema"):
        validate_dataset_job_page({**page, "items": [{"ball_speed": 170.0}]})


def test_player_covariation_payload_enforces_inline_limit_and_identity() -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    payload = build_player_covariation_payload(
        golden["player_covariation_request"]["records"],
        player_column="player_id",
        x_column="face_angle",
        y_column="club_path",
        min_samples=4,
        confidence_level=0.95,
    )
    assert payload == golden["player_covariation_request"]
    with pytest.raises(ValueError, match="20,000"):
        build_player_covariation_payload(
            [{}] * (MAX_CANONICAL_INLINE_RECORDS + 1),
            player_column="player_id",
            x_column="face_angle",
            y_column="club_path",
            min_samples=4,
            confidence_level=0.95,
        )


def test_player_covariation_response_rejects_unsafe_or_unbacked_results() -> None:
    response = {
        "contract_version": "launch-monitor-player-covariation/1.0.0",
        "analysis_kind": "selected_pair",
        "status": "available",
        "request": {},
        "pooled": {},
        "within_player": {},
        "between_player": {},
        "per_player": [],
        "meta_analysis": {},
        "missingness": {},
        "units": {},
        "lineage": {"backing_records": []},
        "availability": [],
        "uncertainty": {},
        "player_identity": {
            "trust_level": "explicit_user_attested",
            "identifier_column": "player_id",
        },
        "vendor_provenance": [],
        "claims": {
            "device_emulation": False,
            "device_certification": False,
            "causal_inference": False,
        },
        "definitions": {},
        "warnings": [],
    }
    assert validate_player_covariation_response(response)["status"] == "available"
    unsafe = {**response, "claims": {**response["claims"], "causal_inference": True}}
    with pytest.raises(ValueError, match="claim"):
        validate_player_covariation_response(unsafe)


def test_client_uses_canonical_dataset_and_covariation_routes(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[tuple[str, str]] = []
    status = {
        "contract_version": "launch-monitor-dataset-job/1.0.0",
        "job_id": "a" * 32,
        "status": "queued",
        "submitted_at_utc": "2026-08-21T00:00:00Z",
        "completed_at_utc": None,
        "input_row_count": 0,
        "result_item_count": 0,
        "unavailable": None,
    }

    class Response:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(status).encode("utf-8")

    def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        assert timeout == 30.0
        calls.append((request.method, request.full_url))
        return Response()

    monkeypatch.setattr(
        "rate_of_closure.launch_monitor_v2_client.urlopen", fake_urlopen
    )
    client = UpstreamV2Client("https://authority.example")
    client.submit_dataset_job({"contract_version": "test"})
    client.dataset_job_status("a" * 32)

    assert calls == [
        (
            "POST",
            "https://authority.example/tools/launch-monitor-analytics/v2/dataset-jobs",
        ),
        (
            "GET",
            "https://authority.example/tools/launch-monitor-analytics/v2/dataset-jobs/"
            + "a" * 32,
        ),
    ]
