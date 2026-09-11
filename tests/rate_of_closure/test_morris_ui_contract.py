"""UI-neutral Morris request, response, client, and presentation contracts."""

from __future__ import annotations

import json
import math
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
    parse_morris_report,
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
from shared.python.swing_sim.types import PendulumParameters, PlaneOrientation

_REPO_ROOT = Path(__file__).parents[2].resolve()

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


def _r136_base_config() -> SimulationConfig:
    """The authority base mirrored by ``morrisAuthorityRequest.test.ts``."""
    return SimulationConfig(
        scenario=ImpactScenario(113.0),
        club=CLUB_LIBRARY["Driver 10.5°"],
        ball_setup=BallSetup(BallSupportMode.TEE, 0.0381),
        source_kind="double_pendulum",
        contact_mode=ContactMode.FIXED_BALL_CONTACT,
        swing_duration_s=1.0,
        plane=PlaneOrientation(0.0, -45.0, 0.0),
        pendulum_parameters=PendulumParameters(
            m1=4.0,
            l1=0.65,
            lc1=0.3,
            i1=0.4,
            m2=0.5,
            l2=1.05,
            lc2=0.55,
            i2=0.08,
            d1=0.4,
            d2=0.25,
        ),
    )


def test_r136_suggested_drafts_match_the_shared_python_artifact() -> None:
    """Anchor the R13.6 base-centered suggestions to a shared fixture.

    The TypeScript authority-request gate compares its computed drafts
    against this same fixture (#4458), so neither runtime can drift from
    the Python registry-derived bounds without failing exactly one gate.
    """
    fixture = json.loads(
        (
            _REPO_ROOT
            / "src"
            / "rate_of_closure"
            / "web"
            / "src"
            / "model"
            / "__fixtures__"
            / "morris_suggested_factor_drafts_v1.json"
        ).read_text(encoding="utf-8")
    )
    drafts = suggested_factor_drafts(_r136_base_config())

    assert [
        {
            "variable_key": draft.variable_key,
            "enabled": draft.enabled,
            "lower": draft.lower,
            "upper": draft.upper,
        }
        for draft in drafts
    ] == fixture["drafts"]


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


def _morris_linear_oracle(
    coefficient: float, lower: float, upper: float
) -> tuple[float, float, float, float]:
    """Analytical closed-form oracle for Morris elementary effects of a linear term.

    For f(x) = beta * x with factor bounds [lower, upper] and span S = upper - lower:
    - Elementary effect EE = beta * S across all trajectories and grid steps.
    - mu = beta * S
    - mu* = |beta| * S
    - sigma = 0.0
    - SE(mu*) = 0.0
    """
    assert lower < upper
    span = upper - lower
    return coefficient * span, abs(coefficient) * span, 0.0, 0.0


def _morris_polynomial_quadratic_oracle(
    coefficient: float, valid_pairs: int = 12
) -> tuple[float, float, float, float]:
    """Analytical closed-form oracle for Morris elementary effects of f(w) = c * w^2.

    On a 4-level balanced grid design with normalized step Delta = 2/3:
    Elementary effects evaluate to 2/3 * c and 4/3 * c in equal proportions
    across trajectories.
    - mu = c
    - mu* = |c|
    - sigma = sqrt((valid_pairs / (valid_pairs - 1)) * (c / 3)^2)
    - SE = sigma / sqrt(valid_pairs)
    """
    assert valid_pairs >= 2
    variance = (valid_pairs / (valid_pairs - 1)) * (coefficient / 3.0) ** 2
    sigma = math.sqrt(variance)
    standard_error = sigma / math.sqrt(valid_pairs)
    return coefficient, abs(coefficient), sigma, standard_error


def test_morris_analytical_oracle_ground_truth_matches_closed_form() -> None:
    fixture_path = (
        _REPO_ROOT
        / "src"
        / "rate_of_closure"
        / "web"
        / "src"
        / "model"
        / "__fixtures__"
        / "morris_analytical_ground_truth_v1.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture["schema_id"] == "rate-of-closure/morris-analytical-ground-truth"
    assert fixture["schema_version"] == 1

    # Verify linear oracle matches expected closed form exactly:
    mu, mu_star, sigma, se = _morris_linear_oracle(-3.5, -15.0, 5.0)
    assert mu == -70.0
    assert mu_star == 70.0
    assert sigma == 0.0
    assert se == 0.0

    mu, mu_star, sigma, se = _morris_linear_oracle(2.5, -2.0, 8.0)
    assert mu == 25.0
    assert mu_star == 25.0
    assert sigma == 0.0
    assert se == 0.0

    # Verify quadratic oracle matches expected closed form exactly:
    mu, mu_star, sigma, se = _morris_polynomial_quadratic_oracle(15.0, 12)
    assert mu == 15.0
    assert mu_star == 15.0
    assert sigma == pytest.approx(5.222329678670935, rel=1e-12)
    assert se == pytest.approx(1.5075567228888122, rel=1e-12)

    for factor in fixture["analytical_factors"]:
        if factor["model_type"] == "linear":
            d_mu, d_mu_star, d_sigma, d_se = _morris_linear_oracle(
                factor["coefficient"], factor["lower"], factor["upper"]
            )
            assert d_mu == pytest.approx(factor["expected_mu"], rel=1e-12)
            assert d_mu_star == pytest.approx(factor["expected_mu_star"], rel=1e-12)
            assert d_sigma == pytest.approx(factor["expected_sigma"], abs=1e-12)
            assert d_se == pytest.approx(factor["expected_standard_error"], abs=1e-12)
        elif factor["model_type"] == "polynomial_quadratic":
            d_mu, d_mu_star, d_sigma, d_se = _morris_polynomial_quadratic_oracle(
                factor["coefficient"], 12
            )
            assert d_mu == pytest.approx(factor["expected_mu"], rel=1e-12)
            assert d_mu_star == pytest.approx(factor["expected_mu_star"], rel=1e-12)
            assert d_sigma == pytest.approx(factor["expected_sigma"], rel=1e-12)
            assert d_se == pytest.approx(factor["expected_standard_error"], rel=1e-12)
        else:
            assert factor["expected_mu"] == 0.0
            assert factor["expected_mu_star"] == 0.0
            assert factor["expected_sigma"] == 0.0
            assert factor["expected_standard_error"] == 0.0


def test_morris_analytical_oracle_presentation_and_scale_awareness() -> None:
    from rate_of_closure.application.morris._metric_validation import (
        validate_finite_metrics,
    )

    fixture_path = (
        _REPO_ROOT
        / "src"
        / "rate_of_closure"
        / "web"
        / "src"
        / "model"
        / "__fixtures__"
        / "morris_analytical_ground_truth_v1.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    job = parse_morris_job(fixture["completed_job"])
    assert job.report is not None

    presentation = present_morris_report(job.report, "clubhead_x_m")
    assert (
        json.loads(json.dumps(asdict(presentation)))
        == fixture["expected_presentation_table"]
    )

    # Assert ranking strictly follows closed-form mu*:
    # Rank 1: side_tilt (mu* = 70.0)
    # Rank 2: yaw (mu* = 25.0)
    # Rank 3: shoulder_damping (mu* = 15.0)
    # Rank 4: wrist_damping (mu* = 0.0)
    assert [(row.rank, row.spec_id, row.mu_star) for row in presentation.rows] == [
        (1, "swing-side-tilt", 70.0),
        (2, "swing-yaw", 25.0),
        (3, "shoulder-damping", 15.0),
        (4, "wrist-damping", 0.0),
    ]

    # Reject scale-blindness / per-physical-unit convention:
    # Bare coefficients without span scaling would be 3.5, 2.5, which would
    # incorrectly promote shoulder-damping (15.0) to rank 1.
    assert presentation.rows[0].mu_star != 3.5
    assert presentation.rows[1].mu_star != 2.5

    # Reject missing step divisor (1/Delta) bug:
    unnormalized = [70.0 * (2 / 3), 25.0 * (2 / 3)]
    assert presentation.rows[0].mu_star != pytest.approx(unnormalized[0], rel=1e-6)
    assert presentation.rows[1].mu_star != pytest.approx(unnormalized[1], rel=1e-6)

    # Negative signed mu correctly preserved alongside positive mu*:
    assert presentation.rows[0].mu == -70.0
    assert presentation.rows[0].mu_star == 70.0

    # Metric invariant holds across all estimates:
    for estimate in job.report.estimates:
        validate_finite_metrics(
            estimate.effects,
            estimate.availability,
            estimate.denominator.valid_pairs,
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
            lambda job: job["report"]["estimates"][0]["effects"].update(
                mu=5.0, mu_star=2.0
            ),
            "magnitudes",
        ),
        (
            lambda job: job["report"]["estimates"][0]["effects"].update(sigma=-1.0),
            "magnitudes",
        ),
        (
            lambda job: job["report"]["estimates"][0]["effects"].update(
                mu_star_standard_error=-0.5
            ),
            "magnitudes",
        ),
        (
            lambda job: job["report"]["estimates"][0]["effects"].update(sigma=1e-14),
            "producer zero clamp",
        ),
        (
            lambda job: job["report"]["estimates"][0]["effects"].update(
                mu=1.5, mu_star=1.5, mu_star_standard_error=0.0, sigma=1e-8
            ),
            "clamp-scale degeneracy",
        ),
        (
            lambda job: job["report"]["estimates"][0]["effects"].update(
                mu=1e308, mu_star=1e308, mu_star_standard_error=0.0, sigma=0.0
            ),
            "safely squared",
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


def test_parse_morris_report_accepts_valid_and_rejects_malformed_report() -> None:
    valid_report = _report()
    parsed = parse_morris_report(valid_report)
    assert parsed.trajectories == 4
    assert parsed.levels == 4
    assert len(parsed.estimates) == 4

    # Rejection of invalid schema
    invalid_schema = deepcopy(valid_report)
    invalid_schema["schema_id"] = "invalid-schema"
    with pytest.raises(ValueError, match="schema"):
        parse_morris_report(invalid_schema)

    # Rejection of invalid metric invariant
    invalid_effects = deepcopy(valid_report)
    invalid_effects["estimates"][0]["effects"]["mu_star"] = -2.0  # type: ignore[index]
    with pytest.raises(ValueError, match="magnitudes"):
        parse_morris_report(invalid_effects)
