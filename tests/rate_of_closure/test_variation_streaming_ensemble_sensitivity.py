"""One-at-a-time sensitivity over strict durable ensemble archives."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rate_of_closure.simulation import ContactMode
from rate_of_closure.variation.durable_ensemble_chunks import DurableEnsembleChunkSink
from rate_of_closure.variation.request_builder import build_simulation_ensemble_request
from rate_of_closure.variation.simulation_adapter import (
    run_simulation_ensemble,
    run_simulation_ensemble_chunks,
)
from rate_of_closure.variation.streaming_ensemble_sensitivity import (
    DurableSensitivityStudy,
    analyze_durable_oat_sensitivity,
)
from shared.python.swing_sim.variation import (
    CATEGORY_SWING,
    NoiseSpec,
    VariationPlan,
    finite_sample_standard_deviation,
    sensitivity_from_standard_deviations,
)

from .test_variation_simulation_adapter import _config

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _plan() -> VariationPlan:
    return VariationPlan(
        mode="swing",
        noise=(
            NoiseSpec(f"{CATEGORY_SWING}.yaw_deg", scale=0.2),
            NoiseSpec(f"{CATEGORY_SWING}.damping_wrist", scale=0.02),
        ),
        n_runs=3,
        seed=17,
    )


def test_durable_oat_sensitivity_matches_materialized_substudies(
    tmp_path: Path,
) -> None:
    plan = _plan()
    base = _config(ContactMode.DELIVERY_INSPECTION)
    studies: dict[str, DurableSensitivityStudy] = {}
    expected_rows: list[np.ndarray] = []
    output_names: tuple[str, ...] | None = None
    for spec in plan.noise:
        sub_plan = replace(plan, noise=(spec,), groups=())
        source = build_simulation_ensemble_request(sub_plan, base)
        directory = tmp_path / str(spec.spec_id)
        run_simulation_ensemble_chunks(
            source, DurableEnsembleChunkSink(directory), chunk_size=2
        )
        studies[spec.variable_key] = DurableSensitivityStudy(source, directory)
        materialized = run_simulation_ensemble(source).variation
        output_names = materialized.output_names
        row = np.full(len(output_names), np.nan)
        for index in range(len(output_names)):
            values = materialized.outputs[:, index]
            finite = values[np.isfinite(values)]
            if finite.size >= 2:
                row[index] = finite_sample_standard_deviation(finite)
        expected_rows.append(row)
    assert output_names is not None
    expected = sensitivity_from_standard_deviations(
        tuple(spec.variable_key for spec in plan.noise),
        output_names,
        np.vstack(expected_rows),
    )

    actual = analyze_durable_oat_sensitivity(plan, studies)

    assert tuple(item.status for item in actual.archives) == ("complete", "complete")
    assert actual.result.input_keys == expected.input_keys
    assert actual.result.output_names == expected.output_names
    np.testing.assert_allclose(actual.result.matrix, expected.matrix, equal_nan=True)
    np.testing.assert_allclose(
        actual.result.normalized, expected.normalized, equal_nan=True
    )
