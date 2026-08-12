"""Python authority for the regional scalar-ensemble import golden fixture."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from rate_of_closure.variation.regional_ground_study_adapter import (
    RegionalGroundStudyOutcome,
)
from rate_of_closure.variation.regional_ground_variation import (
    GroundRegionalVariationTrial,
    register_ground_variation_variables,
    run_regional_ground_variation,
)
from shared.python.swing_sim.flight import execute_regional_ground_from_flight
from shared.python.swing_sim.flight.tests._regional_ground_pipeline_support import (
    _crossing_result,
    _launch,
    _settings,
)
from shared.python.swing_sim.ground import (
    RegionalGroundExecutionOptions,
    SkidRollSettings,
)
from shared.python.swing_sim.variation import registry as variation_registry
from tests.rate_of_closure.regional_ground_target_support import transfer_failure
from tests.rate_of_closure.test_regional_ground_variation import _request

FIXTURE = (
    Path(__file__).parents[2]
    / "src/rate_of_closure/web/src/model/__fixtures__"
    / "regional_ground_scalar_ensemble_golden_v1.json"
)
FIXTURE_SHA256 = "33deb04ec379d090b6711bd64d4daaf6db2d7c8d7d5c51a221453aac51bd6e58"


def _execute_fixture_trial(
    trial: GroundRegionalVariationTrial,
) -> RegionalGroundStudyOutcome:
    if trial.trial_index == 3:
        return transfer_failure()
    settings = _settings(max_time_s=0.35 if trial.trial_index == 1 else 12.0)
    settings = replace(
        settings,
        surface=replace(
            settings.surface,
            rolling_resistance=trial.regional_plan.base_surface.rolling_resistance,
            normal_restitution=trial.regional_plan.base_surface.normal_restitution,
        ),
    )
    options = (
        RegionalGroundExecutionOptions(settings=SkidRollSettings(max_steps=1))
        if trial.trial_index == 2
        else None
    )
    return execute_regional_ground_from_flight(
        _crossing_result(),
        _launch(),
        settings,
        trial.regional_plan,
        capture_speed_m_s=3.0,
        options=options,
    )


def test_python_producer_exactly_recreates_shared_result_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolated = dict(variation_registry.variable_registry())
    monkeypatch.setattr(variation_registry, "_REGISTRY", isolated)
    register_ground_variation_variables()

    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    actual = run_regional_ground_variation(_request(), _execute_fixture_trial).to_wire()
    canonical = json.dumps(actual, sort_keys=True, separators=(",", ":"))

    assert actual == expected
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == FIXTURE_SHA256
    assert [row["trial_index"] for row in actual["rows"]] == [0, 1, 2, 3]
    assert [row["cohort"] for row in actual["rows"]] == [
        "complete",
        "partial",
        "failed",
        "unavailable",
    ]
