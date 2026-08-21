"""Contracts for the cross-runtime registered-run trace catalog."""

from __future__ import annotations

import json
from hashlib import sha256
from importlib import resources

import numpy as np
import pytest

from shared.python.swing_sim.canonical_numeric_json import canonical_numeric_json
from shared.python.swing_sim.rotating_base import (
    EXPECTED_RUN_CATALOG_SHA256,
    EXPECTED_STUDY_SHA256,
    EXPECTED_UPSTREAM_SOURCE_REVISION,
    RotatingBaseRunRequest,
    RotatingBaseRunResult,
    RotatingBaseRunTrace,
    load_embedded_qualified_study,
    registered_requests,
    registered_run_catalog_json,
    registered_run_catalog_mapping,
)

_CATALOG_RESOURCE = "rotating_base_registered_runs_v1.json"


def _trace() -> RotatingBaseRunTrace:
    return RotatingBaseRunTrace(
        time_s=np.array([0.0, 0.001]),
        torso_rate_rad_s=np.array([1.0, 1.1]),
        club_rate_rad_s=np.array([2.0, 2.1]),
        clubhead_speed_m_s=np.array([3.0, 3.1]),
        contact_power_on_club_w=np.array([0.0, 1.0]),
        force_generated_couple_nm=np.array([0.0, 2.0]),
        force_on_club_n=np.zeros((2, 2, 2)),
        distal_segment_kinetic_energy_j=np.array([1.0, 2.0]),
    )


def _results() -> tuple[RotatingBaseRunResult, ...]:
    study = load_embedded_qualified_study().study
    return tuple(
        RotatingBaseRunResult(
            request=RotatingBaseRunRequest(
                torso_profile=case.torso_profile,
                matching_rule=case.matching_rule,
                initial_torso_rate_rad_s=case.initial_torso_rate_rad_s,
            ),
            case=case,
            trace=_trace(),
        )
        for case in study.cases
    )


def test_registered_request_order_matches_publication_case_indices() -> None:
    requests = registered_requests()

    assert tuple(request.case_index for request in requests) == tuple(range(18))
    assert requests[0] == RotatingBaseRunRequest(
        "accelerate", "relative_club_rate", 1.5
    )
    assert requests[-1] == RotatingBaseRunRequest(
        "decelerate", "absolute_club_rate", 5.5
    )


def test_catalog_retains_all_traces_adverse_rows_and_authority_pins() -> None:
    payload = registered_run_catalog_mapping(_results())

    assert payload["schema_id"] == "swing-sim/rotating-base-run-catalog"
    assert payload["source_revision"] == EXPECTED_UPSTREAM_SOURCE_REVISION
    assert payload["study_sha256"] == EXPECTED_STUDY_SHA256
    assert payload["attempted_run_count"] == 18
    runs = payload["runs"]
    assert isinstance(runs, list)
    assert [run["request"]["case_index"] for run in runs] == list(range(18))
    assert [run["case"]["case_index"] for run in runs if not run["case"]["valid"]] == [
        6,
        7,
        8,
        15,
        16,
    ]
    encoded = registered_run_catalog_json(_results())
    assert encoded == registered_run_catalog_json(_results())
    assert len(json.loads(encoded)["runs"]) == 18


def test_embedded_catalog_is_canonical_complete_and_digest_pinned() -> None:
    text = (
        resources.files("shared.python.swing_sim.rotating_base")
        .joinpath("resources", _CATALOG_RESOURCE)
        .read_text(encoding="utf-8")
        .rstrip("\n")
    )
    payload = json.loads(text)

    assert sha256(text.encode("utf-8")).hexdigest() == EXPECTED_RUN_CATALOG_SHA256
    assert canonical_numeric_json(payload) == text
    assert payload["source_revision"] == EXPECTED_UPSTREAM_SOURCE_REVISION
    assert payload["study_sha256"] == EXPECTED_STUDY_SHA256
    assert len(payload["runs"]) == 18
    assert all(len(run["trace"]["time_s"]) == 241 for run in payload["runs"])


def test_catalog_fails_closed_on_missing_or_reordered_run() -> None:
    results = _results()

    with pytest.raises(ValueError, match="complete 18-case order"):
        registered_run_catalog_mapping(results[:-1])
    with pytest.raises(ValueError, match="complete 18-case order"):
        registered_run_catalog_mapping(tuple(reversed(results)))
