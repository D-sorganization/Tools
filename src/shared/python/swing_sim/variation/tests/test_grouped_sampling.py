"""Deterministic grouped/correlated perturbation sampling contracts."""

from __future__ import annotations

import numpy as np
import pytest

from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.variation import (
    CATEGORY_DELIVERY,
    NoiseSpec,
    VariationPlan,
    run_variation,
    sample_input_chunks,
    sample_inputs,
)
from shared.python.swing_sim.variation.spec import PerturbationGroup

pytestmark = pytest.mark.physics

_FACE = f"{CATEGORY_DELIVERY}.face_angle_deg"
_SPEED = f"{CATEGORY_DELIVERY}.clubhead_speed_mps"


def _grouped_plan(*, seed: int = 11, matrix_kind: str = "correlation") -> VariationPlan:
    matrix = (
        ((1.0, 0.7), (0.7, 1.0))
        if matrix_kind == "correlation"
        else ((4.0, 1.4), (1.4, 1.0))
    )
    return VariationPlan(
        mode="delivery",
        noise=(
            NoiseSpec(_FACE, scale=2.0, spec_id="face"),
            NoiseSpec(_SPEED, scale=1.0, spec_id="speed"),
        ),
        groups=(
            PerturbationGroup(
                group_id="delivery-group",
                spec_ids=("face", "speed"),
                matrix=matrix,
                matrix_kind=matrix_kind,
            ),
        ),
        n_runs=20_000,
        seed=seed,
    )


def test_same_seed_is_exact_and_different_seed_changes_grouped_samples() -> None:
    first = sample_inputs(_grouped_plan(seed=11))
    repeated = sample_inputs(_grouped_plan(seed=11))
    different = sample_inputs(_grouped_plan(seed=12))

    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, different)


@pytest.mark.parametrize("chunk_size", [1, 17, 513, 20_000])
def test_grouped_lazy_chunks_are_byte_exact_with_eager_sampling(
    chunk_size: int,
) -> None:
    plan = _grouped_plan()

    actual = np.vstack(
        [values for _, values in sample_input_chunks(plan, chunk_size=chunk_size)]
    )

    np.testing.assert_array_equal(actual, sample_inputs(plan))


@pytest.mark.parametrize("matrix_kind", ["correlation", "covariance"])
def test_sampled_correlation_and_scales_match_declared_semantics(
    matrix_kind: str,
) -> None:
    plan = _grouped_plan(matrix_kind=matrix_kind)
    samples = sample_inputs(plan)
    centered = samples - np.array(
        [plan.resolved_base()[_FACE], plan.resolved_base()[_SPEED]]
    )

    assert np.corrcoef(centered, rowvar=False)[0, 1] == pytest.approx(0.7, abs=0.025)
    np.testing.assert_allclose(np.std(centered, axis=0), [2.0, 1.0], rtol=0.025)


def test_independent_spec_stream_remains_subset_stable_with_explicit_ids() -> None:
    both = VariationPlan(
        mode="delivery",
        noise=(
            NoiseSpec(_FACE, scale=2.0, spec_id="face-stream"),
            NoiseSpec(_SPEED, scale=1.0, spec_id="speed-stream"),
        ),
        n_runs=128,
        seed=3,
    )
    face_only = VariationPlan(
        mode="delivery",
        noise=(NoiseSpec(_FACE, scale=2.0, spec_id="face-stream"),),
        n_runs=128,
        seed=3,
    )

    np.testing.assert_array_equal(
        sample_inputs(both)[:, 0], sample_inputs(face_only)[:, 0]
    )


def test_localized_metadata_is_rejected_by_scalar_execution_capability_check() -> None:
    plan = VariationPlan(
        mode="delivery",
        noise=(
            NoiseSpec(
                _FACE,
                scale=1.0,
                spec_id="face-at-impact",
                time_window_s=(0.7, 0.8),
                point_ids=("swing.clubhead",),
            ),
        ),
        n_runs=4,
        seed=2,
    )

    sample_inputs(plan)
    with pytest.raises(ContractViolationError, match="global perturbations"):
        run_variation(plan)


def test_grouped_run_remains_worker_count_invariant() -> None:
    plan = _grouped_plan()
    plan = VariationPlan(
        mode=plan.mode,
        noise=plan.noise,
        groups=plan.groups,
        n_runs=8,
        seed=plan.seed,
    )

    serial = run_variation(plan, n_workers=1)
    parallel = run_variation(plan, n_workers=4)

    np.testing.assert_array_equal(serial.inputs, parallel.inputs)
    np.testing.assert_array_equal(serial.outputs, parallel.outputs)
