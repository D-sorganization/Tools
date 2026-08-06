"""End-to-end adapter tests for retained chip-shot forgiveness trials."""

from __future__ import annotations

import json
import threading
from dataclasses import replace

import numpy as np
import pytest

import rate_of_closure.variation.forgiveness_runner as forgiveness_runner
from rate_of_closure.club import get_club
from rate_of_closure.model import ImpactScenario
from rate_of_closure.simulation import SimulationConfig, SimulationRun, run_simulation
from rate_of_closure.variation import SimulationEnsembleRequest, run_simulation_ensemble
from rate_of_closure.variation.chip_forgiveness import ChipTrialCohort
from rate_of_closure.variation.forgiveness_io import (
    chip_forgiveness_study_to_csv,
    chip_forgiveness_study_to_json,
)
from rate_of_closure.variation.forgiveness_projection import (
    forgiveness_variation_dataset,
)
from rate_of_closure.variation.forgiveness_runner import (
    ChipForgivenessRequest,
    ChipLossModel,
    analyze_chip_forgiveness_ensemble,
    run_chip_forgiveness_study,
)
from rate_of_closure.variation.plot_labels import OUTPUT_LABELS, OUTPUT_UNITS
from shared.python.golf_club import (
    GroundPlane,
    TurfPreset,
    WedgePreset,
    turf_profile_preset,
    wedge_preset,
)
from shared.python.swing_sim.solver.solve import CancelledError
from shared.python.swing_sim.variation import (
    CATEGORY_DELIVERY,
    NoiseSpec,
    VariationPlan,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _ensemble(configs: tuple[SimulationConfig, ...]) -> SimulationEnsembleRequest:
    variable = f"{CATEGORY_DELIVERY}.face_angle_deg"
    plan = VariationPlan(
        mode="delivery",
        noise=(NoiseSpec(variable, scale=1.0),),
        n_runs=len(configs),
        seed=17,
    )
    samples = np.arange(len(configs), dtype=float).reshape(-1, 1)
    return SimulationEnsembleRequest(plan, samples, configs)


def _request(configs: tuple[SimulationConfig, ...]) -> ChipForgivenessRequest:
    return ChipForgivenessRequest(
        candidate_id="mid-bounce-56",
        ensemble=_ensemble(configs),
        wedge_parameters=wedge_preset(WedgePreset.MID_BOUNCE),
        ground=GroundPlane(point_m=(0.0, -10.0, 0.0)),
        turf_profile=turf_profile_preset(TurfPreset.FIRM_FAIRWAY),
        loss_model=ChipLossModel(target_carry_m=27.432),
        bootstrap_samples=128,
    )


def _config(speed_mph: float) -> SimulationConfig:
    return SimulationConfig(
        scenario=ImpactScenario(clubhead_speed_mph=speed_mph),
        club=get_club("Sand Wedge"),
        impact_time_s=0.03,
    )


def test_runner_retains_success_and_failure_with_advanced_metrics() -> None:
    request = _request((_config(30.0), _config(31.0)))
    progress: list[int] = []

    def executor(config: SimulationConfig) -> SimulationRun:
        if config.scenario.clubhead_speed_mph == 31.0:
            raise RuntimeError("planted integration failure")
        return run_simulation(config)

    study = run_chip_forgiveness_study(
        request,
        executor=executor,
        progress_cb=lambda report: progress.append(report.iteration),
    )

    hit, failure = study.records
    assert hit.cohort is ChipTrialCohort.BALL_ONLY
    assert hit.metrics["carry_m"] is not None
    assert hit.metrics["shaft_rotation_rate_rad_s"] is not None
    assert hit.metrics["shaft_counterfactual_aoa_delta_deg"] is not None
    assert hit.metrics["ground_after_ball_margin_s"] is None
    assert failure.cohort is ChipTrialCohort.NUMERICAL_FAILURE
    assert failure.loss == pytest.approx(request.loss_model.numerical_failure_penalty)
    assert failure.constraint_violated is True
    assert failure.diagnostic == "RuntimeError: planted integration failure"
    assert study.summary.sample_count == 2
    assert study.summary.expected_loss == pytest.approx((hit.loss + failure.loss) / 2.0)
    assert progress == [1, 2]


def test_runner_replay_is_deterministic_for_the_same_seed() -> None:
    request = _request((_config(30.0), _config(30.0)))

    first = run_chip_forgiveness_study(request)
    second = run_chip_forgiveness_study(request)

    assert first.records == second.records
    assert first.summary.expected_loss_ci_low == second.summary.expected_loss_ci_low
    assert first.summary.expected_loss_ci_high == second.summary.expected_loss_ci_high


def test_retained_ensemble_analysis_avoids_reexecuting_trials() -> None:
    request = _request((_config(30.0), _config(30.0)))
    ensemble = run_simulation_ensemble(request.ensemble)

    study = analyze_chip_forgiveness_ensemble(request, ensemble)

    assert len(study.records) == 2
    assert all(record.cohort is ChipTrialCohort.BALL_ONLY for record in study.records)
    assert study.summary.sample_count == 2
    payload = json.loads(chip_forgiveness_study_to_json(study))
    assert payload["schema_version"] == 1
    assert payload["metadata"]["seed"] == request.ensemble.plan.seed
    assert payload["metadata"]["sampling_design"] == "iid-monte-carlo-joint"
    assert payload["records"][0]["turf_contact_status"] is None
    assert len(payload["physics_inputs"]["simulation_configs"]) == 2
    assert payload["physics_inputs"]["loss_model"]["target_carry_m"] == pytest.approx(
        27.432
    )
    assert len(payload["records"]) == 2
    csv_text = chip_forgiveness_study_to_csv(study)
    assert len(csv_text.splitlines()) == 3
    assert "cohort" in csv_text.splitlines()[0]
    projected = forgiveness_variation_dataset(study)
    assert "loss" in projected.output_names
    assert "shaft_counterfactual_aoa_delta_deg" in projected.output_names
    assert projected.outputs.shape[0] == 2
    assert set(projected.output_names) <= OUTPUT_LABELS.keys()
    assert set(projected.output_names) <= OUTPUT_UNITS.keys()


def test_runner_honors_cancellation_before_any_trial() -> None:
    cancellation = threading.Event()
    cancellation.set()

    with pytest.raises(CancelledError):
        run_chip_forgiveness_study(
            _request((_config(30.0),)), cancel_event=cancellation
        )


def test_retained_analysis_honors_cancellation_before_postprocessing() -> None:
    request = _request((_config(30.0),))
    ensemble = run_simulation_ensemble(request.ensemble)
    cancellation = threading.Event()
    cancellation.set()

    with pytest.raises(CancelledError):
        analyze_chip_forgiveness_ensemble(
            request,
            ensemble,
            cancel_event=cancellation,
        )


def test_retained_postprocessing_failure_remains_one_all_trial_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request((_config(30.0), _config(30.0)))
    ensemble = run_simulation_ensemble(request.ensemble)

    def fail_postprocessing(*_args: object) -> object:
        raise ValueError("planted post-processing failure")

    monkeypatch.setattr(forgiveness_runner, "_trial_metrics", fail_postprocessing)
    study = analyze_chip_forgiveness_ensemble(request, ensemble)

    assert study.summary.sample_count == 2
    assert all(
        record.cohort is ChipTrialCohort.NUMERICAL_FAILURE for record in study.records
    )
    assert all(
        "planted post-processing failure" in (record.diagnostic or "")
        for record in study.records
    )


def test_loss_contract_penalizes_ground_first_and_declared_constraints() -> None:
    model = ChipLossModel(
        target_carry_m=10.0,
        carry_tolerance_m=2.0,
        lateral_tolerance_m=1.0,
        maximum_turf_penetration_m=0.01,
        include_turf_penetration=True,
    )
    metrics = {
        "carry_m": 12.0,
        "lateral_m": -1.0,
        "ground_after_ball_margin_s": -0.001,
        "peak_turf_penetration_m": 0.02,
    }

    loss, violated = model.evaluate(ChipTrialCohort.GROUND_FIRST, metrics)

    assert violated is True
    assert loss == pytest.approx(model.ground_first_penalty + 1.0 + 1.0 + 4.0)


def test_loss_rejects_unsupported_turf_and_missing_required_outputs() -> None:
    model = ChipLossModel(target_carry_m=10.0)

    loss, violated = model.evaluate(
        ChipTrialCohort.BALL_ONLY,
        {"carry_m": None, "lateral_m": None, "peak_turf_penetration_m": 0.2},
        turf_contact_status="outside_calibrated_domain",
    )

    assert violated is True
    assert loss == pytest.approx(
        model.missing_required_metric_penalty + model.unsupported_turf_penalty
    )


def test_uncalibrated_turf_cannot_enter_the_ranking_loss() -> None:
    request = _request((_config(30.0),))

    with pytest.raises(ValueError, match="calibrated profile"):
        replace(
            request,
            loss_model=ChipLossModel(include_turf_penetration=True),
        )
