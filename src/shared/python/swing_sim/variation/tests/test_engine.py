"""Engine behaviour: seeding, truncation, parallelism, cancel (#4120 V3)."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.variation import (
    CATEGORY_BALL_SETUP,
    CATEGORY_DELIVERY,
    CATEGORY_LAUNCH,
    CATEGORY_SWING,
    CancelledError,
    NoiseSpec,
    VariationPlan,
    outputs_for_mode,
    run_variation,
    sample_input_block,
    sample_input_chunks,
    sample_inputs,
)

pytestmark = pytest.mark.physics

_FACE = f"{CATEGORY_DELIVERY}.face_angle_deg"
_SPEED = f"{CATEGORY_DELIVERY}.clubhead_speed_mps"
_BALL = f"{CATEGORY_LAUNCH}.ball_speed_mph"
_TEE_HEIGHT = f"{CATEGORY_BALL_SETUP}.tee_height_m"


def _delivery_plan(n_runs: int = 8, seed: int = 3) -> VariationPlan:
    return VariationPlan(
        mode="delivery",
        noise=(
            NoiseSpec(_FACE, scale=2.0),
            NoiseSpec(_SPEED, distribution="uniform", scale=1.0),
        ),
        n_runs=n_runs,
        seed=seed,
    )


def _launch_plan(n_runs: int = 16, seed: int = 5) -> VariationPlan:
    return VariationPlan(
        mode="launch",
        noise=(NoiseSpec(_BALL, scale=1.0),),
        n_runs=n_runs,
        seed=seed,
    )


class TestSampling:
    def test_same_seed_same_samples_different_seed_differs(self) -> None:
        a = sample_inputs(_delivery_plan(seed=3))
        b = sample_inputs(_delivery_plan(seed=3))
        c = sample_inputs(_delivery_plan(seed=4))
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, c)

    def test_truncation_is_respected_for_every_distribution(self) -> None:
        for distribution in ("normal", "uniform", "triangular"):
            plan = VariationPlan(
                mode="delivery",
                noise=(
                    NoiseSpec(
                        _FACE,
                        distribution=distribution,
                        scale=5.0,
                        lower=-1.0,
                        upper=1.5,
                    ),
                ),
                n_runs=500,
                seed=11,
            )
            samples = sample_inputs(plan)
            assert float(samples.min()) >= -1.0
            assert float(samples.max()) <= 1.5

    def test_streams_are_subset_stable_per_variable(self) -> None:
        """Dropping one spec leaves the other spec's draws unchanged."""
        both = _delivery_plan(n_runs=32)
        only_face = VariationPlan(
            mode="delivery",
            noise=(NoiseSpec(_FACE, scale=2.0),),
            n_runs=32,
            seed=both.seed,
        )
        np.testing.assert_array_equal(
            sample_inputs(both)[:, 0], sample_inputs(only_face)[:, 0]
        )

    @pytest.mark.parametrize("chunk_size", [1, 3, 11, 64])
    def test_lazy_chunks_are_exactly_eager_and_canonically_ordered(
        self, chunk_size: int
    ) -> None:
        plan = _delivery_plan(n_runs=32, seed=3)

        chunks = tuple(sample_input_chunks(plan, chunk_size=chunk_size))

        assert tuple(start for start, _ in chunks) == tuple(
            range(0, plan.n_runs, chunk_size)
        )
        np.testing.assert_array_equal(
            np.vstack([values for _, values in chunks]), sample_inputs(plan)
        )

    def test_lazy_resume_regenerates_only_the_requested_suffix(self) -> None:
        plan = _delivery_plan(n_runs=32, seed=3)

        chunks = tuple(sample_input_chunks(plan, chunk_size=7, start_index=13))

        assert chunks[0][0] == 13
        np.testing.assert_array_equal(
            np.vstack([values for _, values in chunks]), sample_inputs(plan)[13:]
        )

    def test_exact_lazy_block_handles_a_non_aligned_resume_boundary(self) -> None:
        plan = _delivery_plan(n_runs=32, seed=3)

        block = sample_input_block(plan, start_index=13, row_count=8)

        np.testing.assert_array_equal(block, sample_inputs(plan)[13:21])
        assert not block.flags.writeable

    @pytest.mark.parametrize("distribution", ["normal", "uniform", "triangular"])
    def test_lazy_chunks_preserve_every_marginal_distribution(
        self, distribution: str
    ) -> None:
        plan = VariationPlan(
            mode="delivery",
            noise=(NoiseSpec(_FACE, distribution=distribution, scale=2.0),),
            n_runs=23,
            seed=7,
        )

        actual = np.vstack(
            [values for _, values in sample_input_chunks(plan, chunk_size=4)]
        )

        np.testing.assert_array_equal(actual, sample_inputs(plan))

    @pytest.mark.parametrize("chunk_size", [False, 0, -1, 1.5])
    def test_lazy_chunks_reject_invalid_chunk_size(self, chunk_size: object) -> None:
        with pytest.raises(ContractViolationError, match="positive integer"):
            sample_input_chunks(  # type: ignore[arg-type]
                _delivery_plan(), chunk_size=chunk_size
            )

    @pytest.mark.parametrize("start_index", [False, -1, 9, 1.5])
    def test_lazy_chunks_reject_invalid_start_index(self, start_index: object) -> None:
        with pytest.raises(ContractViolationError, match="within the plan"):
            sample_input_chunks(  # type: ignore[arg-type]
                _delivery_plan(), chunk_size=2, start_index=start_index
            )


class TestRunVariation:
    def test_scalar_engine_rejects_tee_only_context_variable(self) -> None:
        plan = VariationPlan(
            mode="delivery",
            noise=(NoiseSpec(_TEE_HEIGHT, scale=0.002),),
            n_runs=2,
        )

        with pytest.raises(ContractViolationError, match="context-specific"):
            run_variation(plan)

    def test_same_plan_and_seed_gives_identical_dataset(self) -> None:
        plan = _delivery_plan()
        a = run_variation(plan, n_workers=2)
        b = run_variation(plan, n_workers=2)
        np.testing.assert_array_equal(a.inputs, b.inputs)
        np.testing.assert_array_equal(a.outputs, b.outputs)
        np.testing.assert_array_equal(a.success, b.success)

    def test_result_is_worker_count_invariant(self) -> None:
        plan = _launch_plan(n_runs=12)
        serial = run_variation(plan, n_workers=1)
        parallel = run_variation(plan, n_workers=4)
        np.testing.assert_array_equal(serial.outputs, parallel.outputs)

    def test_delivery_mode_populates_all_output_columns(self) -> None:
        dataset = run_variation(_delivery_plan(), n_workers=2)
        assert dataset.output_names == outputs_for_mode("delivery")
        assert dataset.n_success == dataset.plan.n_runs
        carry = dataset.output_column("carry_m")
        assert np.all(np.isfinite(carry)) and np.all(carry > 50.0)

    def test_launch_mode_flies_the_ball(self) -> None:
        dataset = run_variation(_launch_plan(), n_workers=2)
        assert dataset.output_names == outputs_for_mode("launch")
        assert dataset.n_success == dataset.plan.n_runs
        assert float(np.mean(dataset.output_column("carry_m"))) > 100.0

    def test_swing_mode_runs_the_pendulum_pipeline(self) -> None:
        plan = VariationPlan(
            mode="swing",
            base_variables={f"{CATEGORY_SWING}.side_tilt_deg": -45.0},
            noise=(NoiseSpec(f"{CATEGORY_SWING}.yaw_deg", scale=1.0),),
            n_runs=3,
            seed=1,
        )
        dataset = run_variation(plan, n_workers=3)
        assert dataset.n_success == 3
        assert np.all(np.isfinite(dataset.output_column("ball_speed_mph")))

    def test_failed_runs_are_recorded_not_raised(self) -> None:
        """Invalid sampled launch speeds fail their runs, not the batch."""
        plan = VariationPlan(
            mode="launch",
            base_variables={_BALL: 0.5},
            noise=(NoiseSpec(_BALL, scale=3.0),),
            n_runs=64,
            seed=2,
        )
        dataset = run_variation(plan, n_workers=2)
        assert 0 < dataset.n_success < plan.n_runs
        failed = ~dataset.success
        assert np.all(np.isnan(dataset.outputs[failed]))

    def test_progress_reports_arrive_in_solver_shape(self) -> None:
        reports: list[object] = []
        run_variation(_launch_plan(n_runs=16), n_workers=2, progress_cb=reports.append)
        assert reports
        last = reports[-1]
        assert last.iteration == 16  # type: ignore[attr-defined]
        assert last.elapsed_s >= 0.0  # type: ignore[attr-defined]


class TestCancellation:
    def test_pre_set_cancel_event_raises_immediately(self) -> None:
        event = threading.Event()
        event.set()
        with pytest.raises(CancelledError):
            run_variation(_launch_plan(), cancel_event=event)

    def test_mid_run_cancel_raises(self) -> None:
        event = threading.Event()

        def cancel_after_first_report(_report: object) -> None:
            event.set()

        with pytest.raises(CancelledError):
            run_variation(
                _launch_plan(n_runs=256),
                n_workers=2,
                progress_cb=cancel_after_first_report,
                cancel_event=event,
            )
