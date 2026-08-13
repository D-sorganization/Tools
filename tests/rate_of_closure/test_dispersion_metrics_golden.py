"""Python-authority verification for the cross-runtime dispersion fixture."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from shared.python.swing_sim.variation import (
    CATEGORY_SWING,
    EnsemblePositionTraces,
    LowVariabilityMetricCriteria,
    NoiseSpec,
    PositionDispersion,
    VariationDataset,
    VariationPlan,
    build_dispersion_metric_series,
    compute_position_dispersion,
    find_ranked_low_variability_intervals,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_FIXTURE = (
    Path(__file__).parents[2]
    / "src/rate_of_closure/web/src/model/__fixtures__/dispersion_metrics_golden_v1.json"
)
_POINT_ID = "swing.clubhead.reference"


def _authority_dispersion(document: dict[str, object]) -> PositionDispersion:
    traces = document["traces"]
    assert isinstance(traces, list)
    times = np.asarray(document["times_s"], dtype=float)
    positions = np.asarray([item["points_m"] for item in traces], dtype=float)
    yaw = f"{CATEGORY_SWING}.yaw_deg"
    plan = VariationPlan(
        mode="swing",
        noise=(NoiseSpec(yaw, scale=0.1),),
        n_runs=len(traces),
        seed=1,
    )
    dataset = VariationDataset(
        plan=plan,
        input_names=(yaw,),
        inputs=np.zeros((len(traces), 1)),
        output_names=("carry_m",),
        outputs=np.zeros((len(traces), 1)),
        success=np.ones(len(traces), dtype=bool),
    )
    ensemble = EnsemblePositionTraces(
        variation=dataset,
        sample_times_s=times,
        coordinate_frame="app_frame:x_target,y_up,z_right",
        point_ids=(_POINT_ID,),
        positions_m=positions[:, :, np.newaxis, :],
        sample_valid=np.ones(positions.shape[:2], dtype=bool),
        impact_sample_indices=np.full(len(traces), -1, dtype=int),
    )
    return compute_position_dispersion(ensemble)


def test_golden_fixture_is_emitted_by_python_authority() -> None:
    document = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    dispersion = _authority_dispersion(document)
    confidence = float(document["confidence_level"])

    for metric, expected in document["expected"].items():
        series = build_dispersion_metric_series(
            dispersion, _POINT_ID, metric, confidence
        )
        np.testing.assert_allclose(series.values, expected["values"], rtol=5e-13)
        assert list(series.adequacy) == expected["adequacy"]
        criteria = LowVariabilityMetricCriteria(
            metric=metric,
            max_value=expected["threshold"],
            confidence_level=confidence,
            point_ids=(_POINT_ID,),
        )
        intervals = find_ranked_low_variability_intervals(dispersion, criteria)
        assert len(intervals) == len(expected["intervals"])
        for interval, expected_interval in zip(
            intervals, expected["intervals"], strict=True
        ):
            assert interval.start_index == expected_interval["start_index"]
            assert interval.end_index == expected_interval["end_index"]
            assert interval.rank == expected_interval["rank"]
            assert interval.score == pytest.approx(expected_interval["score"])
