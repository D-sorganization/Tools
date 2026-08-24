"""Plan-evidence persistence for chip-forgiveness exports."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rate_of_closure.variation.chip_forgiveness import (
    ChipStudyMetadata,
    ChipTrialCohort,
    ChipTrialRecord,
    summarize_chip_trials,
)
from rate_of_closure.variation.forgiveness_io import chip_forgiveness_study_to_dict
from rate_of_closure.variation.forgiveness_runner import ChipForgivenessStudy
from shared.python.swing_sim.variation import CATEGORY_LAUNCH, NoiseSpec, VariationPlan

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _study() -> ChipForgivenessStudy:
    plan = VariationPlan(
        mode="launch",
        noise=(NoiseSpec(f"{CATEGORY_LAUNCH}.launch_angle_deg", scale=1.0),),
        n_runs=1,
        seed=41,
    )
    record = ChipTrialRecord(0, ChipTrialCohort.BALL_FIRST, 1.0, False)
    metadata = ChipStudyMetadata(
        candidate_id="wedge-a",
        plan_schema="swing-sim.variation-plan/v2",
        coordinate_frame="app_frame:x_target,y_up,z_right",
        seed=41,
        noise_model_id="test-noise-v1",
        objective_id="test-objective-v1",
        turf_profile_id="illustrative-test",
        turf_calibration_status="illustrative",
        solver_id="rate-of-closure/test",
        sampling_design="deterministic-test",
        inference_method_id="test-only",
        limitations="Model-scenario persistence test only.",
    )
    summary = summarize_chip_trials(metadata, (record,), bootstrap_samples=64)
    request = SimpleNamespace(
        ensemble=SimpleNamespace(configs=()),
        wedge_parameters={},
        ground={},
        turf_profile={},
        loss_model={},
        cvar_tail_fraction=0.1,
        bootstrap_samples=64,
    )
    study = object.__new__(ChipForgivenessStudy)
    object.__setattr__(study, "records", (record,))
    object.__setattr__(study, "summary", summary)
    object.__setattr__(study, "plan", plan)
    object.__setattr__(study, "input_names", ())
    object.__setattr__(study, "sampled_inputs", np.empty((1, 0)))
    object.__setattr__(study, "request", request)
    return study


def test_export_embeds_one_cohesive_execution_document() -> None:
    document = chip_forgiveness_study_to_dict(_study())

    assert document["schema_version"] == 2
    assert "plan" not in document
    assert document["plan_document"]["schema_version"] == 3
    assert document["plan_document"]["metadata"]["plan_sha256"]
    assert document["plan_document"]["plan"]["seed"] == 41
