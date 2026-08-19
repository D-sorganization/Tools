"""UI-neutral Morris request, response, client, and presentation contracts."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from rate_of_closure.application.morris.contracts import parse_morris_request
from rate_of_closure.application.morris.presentation import (
    present_morris_factor_rows,
    present_morris_job,
    present_morris_report,
)
from rate_of_closure.application.morris.request_document import (
    CANONICAL_MORRIS_FACTOR_KEYS,
    MorrisFactorDraft,
    build_morris_request,
    suggested_factor_drafts,
)
from rate_of_closure.application.morris.response_contract import (
    parse_morris_capability,
    parse_morris_job,
)
from rate_of_closure.club import CLUB_LIBRARY
from rate_of_closure.model import ImpactScenario
from rate_of_closure.simulation import (
    BallSetup,
    BallSupportMode,
    ContactMode,
    SimulationConfig,
)
from shared.python.swing_sim.flight.registry import FlightModelType

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _config(mode: BallSupportMode = BallSupportMode.TEE) -> SimulationConfig:
    return SimulationConfig(
        scenario=ImpactScenario(113.0),
        club=CLUB_LIBRARY["Driver 10.5°"],
        ball_setup=BallSetup(mode, 0.0381 if mode is BallSupportMode.TEE else 0.0),
        source_kind="double_pendulum",
        contact_mode=ContactMode.FIXED_BALL_CONTACT,
        swing_duration_s=0.05,
    )


def _estimate(
    spec: str,
    key: str,
    mu_star: float | None,
    availability: str,
    valid: int,
) -> dict[str, object]:
    constant = availability == "constant-output"
    effect = 0.0 if constant else mu_star
    return {
        "source": {
            "spec_id": spec,
            "variable_key": key,
            "unit": "deg",
            "bounds": [-1.0, 1.0],
            "time_window_s": None,
            "point_ids": [],
        },
        "target": {
            "name": "carry_m",
            "unit": "m",
            "kind": "shot-outcome",
            "time_s": None,
            "point_id": None,
            "coordinate_frame": "app_frame:x_target,y_up,z_right",
        },
        "effects": {
            "mu": effect,
            "mu_star": effect,
            "mu_star_standard_error": None if effect is None else 0.0,
            "sigma": None if effect is None else 0.0,
        },
        "availability": availability,
        "sample_adequacy": "limited" if valid else "insufficient",
        "denominator": {
            "total_pairs": 4,
            "valid_pairs": valid,
            "typed_no_impact_pairs": max(2, 4 - valid),
            "no_impact_unavailable_pairs": 4 - valid,
            "failed_pairs": 0,
            "nonfinite_pairs": 0,
        },
    }


def _report() -> dict[str, object]:
    return {
        "schema_id": "swing-sim/morris-global-sensitivity-report",
        "schema_version": 1,
        "method": "morris-elementary-effects",
        "design": {
            "trajectories": 4,
            "levels": 4,
            "seed": 7,
            "total_samples": 20,
            "normalized_step": 2 / 3,
        },
        "assumptions": ["bounded"],
        "interaction_caveat": "screening only",
        "estimates": [
            _estimate("forward", CANONICAL_MORRIS_FACTOR_KEYS[2], 3.0, "available", 4),
            _estimate("yaw", CANONICAL_MORRIS_FACTOR_KEYS[0], 3.0, "available", 4),
            _estimate(
                "damping",
                CANONICAL_MORRIS_FACTOR_KEYS[3],
                0.0,
                "constant-output",
                4,
            ),
            _estimate(
                "side",
                CANONICAL_MORRIS_FACTOR_KEYS[1],
                None,
                "insufficient-data",
                0,
            ),
        ],
    }


def _job(status: str = "completed") -> dict[str, object]:
    return {
        "schema_id": "rate-of-closure/morris-job",
        "schema_version": 1,
        "job_id": "job-1",
        "request_id": "request-1",
        "status": status,
        "completed_samples": 20 if status == "completed" else 2,
        "total_samples": 20,
        "cancel_requested": False,
        "report": _report() if status == "completed" else None,
        "error": None,
    }


def test_factor_drafts_have_canonical_order_bounds_and_tee_applicability() -> None:
    tee = suggested_factor_drafts(_config())
    ground = suggested_factor_drafts(_config(BallSupportMode.GROUND))
    assert tuple(item.variable_key for item in tee) == CANONICAL_MORRIS_FACTOR_KEYS
    assert (
        tuple(item.variable_key for item in ground) == CANONICAL_MORRIS_FACTOR_KEYS[:-1]
    )
    assert all(item.lower < item.upper for item in tee)
    assert tee[-1].enabled and tee[-1].variable_key.endswith("tee_height_m")
    for config in (_config(), _config(BallSupportMode.GROUND)):
        request = build_morris_request(
            config,
            suggested_factor_drafts(config),
            request_id="all-suggested",
        )
        assert request.base_config() == config


def test_request_builder_round_trips_full_config_and_is_non_mutating() -> None:
    config = _config()
    drafts = suggested_factor_drafts(config)[:2]
    request = build_morris_request(
        config,
        drafts,
        request_id="request-1",
        trajectories=4,
        levels=4,
        seed=7,
        minimum_effects=2,
        worker_count=1,
    )
    assert request.base_config() == config
    assert parse_morris_request(request.to_json_dict()) == request
    assert config == _config()
    assert drafts == suggested_factor_drafts(config)[:2]


def test_request_builder_serializes_reversed_drafts_in_canonical_order() -> None:
    config = _config()
    drafts = tuple(reversed(suggested_factor_drafts(config)[:2]))
    request = build_morris_request(config, drafts, request_id="ordered")
    assert (
        tuple(factor.variable_key for factor in request.factors)
        == (CANONICAL_MORRIS_FACTOR_KEYS[:2])
    )


def test_python_generated_ui_fixture_is_exact_and_shared() -> None:
    fixture_path = (
        Path(__file__).parents[2]
        / "src/rate_of_closure/web/src/model/__fixtures__/morris_ui_parity_v1.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture["authority_club_names"] == list(CLUB_LIBRARY)
    assert fixture["authority_flight_models"] == [
        model.value for model in FlightModelType
    ]
    config = _config()
    drafts = suggested_factor_drafts(config)[:2]
    request = build_morris_request(
        config,
        drafts,
        request_id="parity-request",
        trajectories=12,
        levels=4,
        seed=73,
        minimum_effects=2,
        worker_count=1,
    )
    job = parse_morris_job(fixture["completed_job"])
    assert fixture["submitted_request"] == request.to_json_dict()
    assert fixture["factor_drafts"] == [asdict(value) for value in drafts]
    assert fixture["expected_factor_rows"] == [
        asdict(value) for value in present_morris_factor_rows(config, drafts)
    ]
    assert fixture["expected_job_presentation"] == asdict(present_morris_job(job))
    assert fixture["expected_tables"]["clubhead_x_m"] == json.loads(
        json.dumps(asdict(present_morris_report(job.report, "clubhead_x_m")))
    )


def test_request_builder_rejects_unrepresented_config_semantics_and_bad_drafts() -> (
    None
):
    config = _config()
    with pytest.raises(ValueError, match="pinned authority"):
        build_morris_request(
            replace(config, impact_time_offset_s=0.001),
            suggested_factor_drafts(config)[:1],
            request_id="request-1",
        )
    bad = (MorrisFactorDraft("swing_sim.swing.yaw_deg", True, 1.0, 1.0),)
    with pytest.raises(ValueError, match="lower < upper"):
        build_morris_request(config, bad, request_id="request-1")
    disabled_bad = (MorrisFactorDraft(CANONICAL_MORRIS_FACTOR_KEYS[0], 0, 0.0, 1.0),)
    with pytest.raises(TypeError, match="enabled"):
        build_morris_request(config, disabled_bad, request_id="request-1")


def test_strict_response_parsers_and_target_scoped_ranking() -> None:
    capability = parse_morris_capability(
        {
            "schema_id": "rate-of-closure/morris-authority-capability",
            "schema_version": 1,
            "available": True,
            "api_prefix": "/api/rate-of-closure/v1",
            "request_schema_id": "rate-of-closure/morris-request",
            "job_schema_id": "rate-of-closure/morris-job",
        }
    )
    assert capability.available
    job = parse_morris_job(_job())
    view = present_morris_report(job.report, "carry_m")
    assert [(row.spec_id, row.rank) for row in view.rows] == [
        ("yaw", 1),
        ("forward", 2),
        ("damping", 3),
        ("side", None),
    ]
    assert view.rows[-1].no_impact_unavailable_pairs == 4
    job_view = present_morris_job(job)
    assert job_view.terminal and job_view.can_present_results
    invalid = _job()
    invalid["extra"] = True
    with pytest.raises(ValueError, match="fields"):
        parse_morris_job(invalid)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda job: job["report"]["design"].update(levels=5),
            "design fields",
        ),
        (
            lambda job: job["report"]["estimates"][0]["source"].update(
                bounds=[1.0, 1.0]
            ),
            "lower < upper",
        ),
        (
            lambda job: job["report"]["estimates"][0]["target"].update(
                kind="state-point", point_id=None
            ),
            "state-point",
        ),
        (
            lambda job: job["report"]["estimates"][0]["effects"].update(mu_star=-1.0),
            "magnitudes",
        ),
        (
            lambda job: job["report"]["estimates"][3]["denominator"].update(
                typed_no_impact_pairs=0
            ),
            "typed no-impact",
        ),
        (
            lambda job: job["report"].update(assumptions=["same", "same"]),
            "unique",
        ),
        (
            lambda job: job.update(completed_samples=19),
            "all samples",
        ),
    ],
)
def test_response_parser_rejects_adversarial_scientific_documents(
    mutate: object, message: str
) -> None:
    document = deepcopy(_job())
    mutate(document)  # type: ignore[operator]
    with pytest.raises((TypeError, ValueError), match=message):
        parse_morris_job(document)
