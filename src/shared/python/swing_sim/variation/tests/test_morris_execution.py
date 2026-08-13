"""UI-neutral execution of deterministic Morris designs (#4142 R13.3)."""

from __future__ import annotations

import threading
from collections.abc import Callable

import numpy as np
import pytest

from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.variation import (
    MAX_MORRIS_OBSERVATION_CELLS,
    MAX_MORRIS_SAMPLES,
    MAX_MORRIS_WORKERS,
    CancelledError,
    MorrisEvaluation,
    MorrisExecutionOptions,
    MorrisFactor,
    MorrisOutput,
    MorrisSample,
    evaluate_morris_design,
    generate_morris_design,
)

pytestmark = pytest.mark.physics


def _factors() -> tuple[MorrisFactor, ...]:
    return (
        MorrisFactor(
            spec_id="face",
            variable_key="swing_sim.impact.delivery.face_angle_deg",
            lower=-2.0,
            upper=4.0,
            unit="deg",
        ),
        MorrisFactor(
            spec_id="speed",
            variable_key="swing_sim.impact.delivery.clubhead_speed_mps",
            lower=40.0,
            upper=55.0,
            unit="m/s",
        ),
    )


def _outputs() -> tuple[MorrisOutput, ...]:
    return (
        MorrisOutput("state", "m", target_kind="scalar"),
        MorrisOutput("impact", "deg", target_kind="impact"),
        MorrisOutput("carry", "m", target_kind="shot-outcome"),
    )


def _design(trajectories: int = 4):
    return generate_morris_design(_factors(), trajectories=trajectories, seed=17)


def _hit(sample: MorrisSample) -> MorrisEvaluation:
    face = sample.physical_values["face"]
    speed = sample.physical_values["speed"]
    return MorrisEvaluation(
        status="evaluated_hit",
        values={"state": face, "impact": speed, "carry": face + speed},
    )


def test_serial_and_parallel_results_are_identical_in_canonical_order() -> None:
    design = _design()

    serial = evaluate_morris_design(
        design, _outputs(), _hit, MorrisExecutionOptions(n_workers=1)
    )
    parallel = evaluate_morris_design(
        design, _outputs(), _hit, MorrisExecutionOptions(n_workers=4)
    )

    np.testing.assert_array_equal(serial.values, parallel.values)
    np.testing.assert_array_equal(serial.outcomes, parallel.outcomes)
    expected = design.physical_points[:, :, 0] + design.physical_points[:, :, 1]
    np.testing.assert_array_equal(serial.values[:, :, 2], expected)


def test_samples_preserve_flattened_identity_scaling_and_immutability() -> None:
    design = _design(trajectories=2)
    seen: list[MorrisSample] = []

    def record(sample: MorrisSample) -> MorrisEvaluation:
        seen.append(sample)
        return _hit(sample)

    evaluate_morris_design(design, _outputs(), record)

    assert [sample.ordinal for sample in seen] == list(range(6))
    assert [(sample.trajectory_index, sample.point_index) for sample in seen] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    ]
    assert seen[0].factors == design.factors
    assert dict(seen[0].physical_values) == dict(
        zip(
            (factor.spec_id for factor in design.factors),
            design.physical_points[0, 0],
            strict=True,
        )
    )
    with pytest.raises(TypeError):
        seen[0].physical_values["face"] = 0.0  # type: ignore[index]


def test_evaluation_mapping_is_immutable() -> None:
    evaluation = _hit(MorrisSample(0, 0, 0, _factors(), {"face": 1.0, "speed": 45.0}))

    with pytest.raises(TypeError):
        evaluation.values["state"] = 0.0  # type: ignore[index]


def test_no_impact_retains_scalar_and_state_point_only() -> None:
    outputs = (
        MorrisOutput("scalar", "m", target_kind="scalar"),
        MorrisOutput(
            "state",
            "m",
            target_kind="state-point",
            target_point_id="clubhead",
            coordinate_frame="app_frame:x_target,y_up,z_right",
        ),
        MorrisOutput("impact", "deg", target_kind="impact"),
    )
    evaluation = MorrisEvaluation(
        "evaluated_no_impact", {"scalar": 1.0, "state": 2.0, "impact": None}
    )

    observations = evaluate_morris_design(
        _design(1), outputs, lambda _sample: evaluation
    )

    np.testing.assert_equal(observations.values[0, 0], (1.0, 2.0, np.nan))


@pytest.mark.parametrize(
    ("evaluation", "expected", "status"),
    [
        (
            MorrisEvaluation(
                "evaluated_hit", {"state": 1.0, "impact": 2.0, "carry": None}
            ),
            (1.0, 2.0, np.nan),
            "evaluated_hit",
        ),
        (
            MorrisEvaluation(
                "evaluated_no_impact",
                {"state": 3.0, "impact": None, "carry": None},
            ),
            (3.0, np.nan, np.nan),
            "evaluated_no_impact",
        ),
        (
            MorrisEvaluation(
                "numerical_failure",
                {"state": None, "impact": None, "carry": None},
            ),
            (np.nan, np.nan, np.nan),
            "numerical_failure",
        ),
    ],
)
def test_typed_statuses_preserve_per_output_availability(
    evaluation: MorrisEvaluation,
    expected: tuple[float, float, float],
    status: str,
) -> None:
    observations = evaluate_morris_design(
        _design(1), _outputs(), lambda _sample: evaluation
    )

    np.testing.assert_equal(observations.values[0, 0], expected)
    assert observations.outcomes[0, 0] == status


def test_evaluator_normalizes_its_domain_failure_explicitly() -> None:
    def fail(_sample: MorrisSample) -> MorrisEvaluation:
        return MorrisEvaluation(
            "numerical_failure",
            {"state": None, "impact": None, "carry": None},
        )

    observations = evaluate_morris_design(_design(1), _outputs(), fail)

    assert np.all(observations.outcomes == "numerical_failure")
    assert np.all(np.isnan(observations.values))


def test_failure_diagnostics_are_retained_without_leaking_to_successes() -> None:
    def evaluate(sample: MorrisSample) -> MorrisEvaluation:
        if sample.ordinal == 1:
            return MorrisEvaluation(
                "numerical_failure",
                {"state": None, "impact": None, "carry": None},
                failure_type="ConvergenceError",
                failure_message="iteration limit reached",
            )
        return MorrisEvaluation(
            "evaluated_hit", {"state": 1.0, "impact": 2.0, "carry": 3.0}
        )

    observations = evaluate_morris_design(_design(1), _outputs(), evaluate)

    assert observations.failure_types.tolist() == [[None, "ConvergenceError", None]]
    assert observations.failure_messages.tolist() == [
        [None, "iteration limit reached", None]
    ]


def test_success_evaluation_rejects_failure_diagnostics() -> None:
    with pytest.raises(ContractViolationError, match="only for numerical failures"):
        MorrisEvaluation(
            "evaluated_hit",
            {"state": 1.0, "impact": 2.0, "carry": 3.0},
            failure_type="Unexpected",
            failure_message="must not survive",
        )


def test_domain_exception_requires_explicit_evaluator_adapter() -> None:
    class DomainError(Exception):
        pass

    def raw_evaluator(_sample: MorrisSample) -> MorrisEvaluation:
        raise DomainError("missed feasible domain")

    with pytest.raises(DomainError, match="feasible domain"):
        evaluate_morris_design(_design(1), _outputs(), raw_evaluator)

    def normalized_evaluator(sample: MorrisSample) -> MorrisEvaluation:
        try:
            return raw_evaluator(sample)
        except DomainError:
            return MorrisEvaluation(
                "numerical_failure",
                {"state": None, "impact": None, "carry": None},
            )

    result = evaluate_morris_design(_design(1), _outputs(), normalized_evaluator)
    assert np.all(result.outcomes == "numerical_failure")


@pytest.mark.parametrize(
    ("evaluation_factory", "message"),
    [
        (
            lambda: MorrisEvaluation("evaluated_hit", {"state": 1.0}),
            "exact output-name set",
        ),
        (
            lambda: MorrisEvaluation(
                "evaluated_hit", {"state": 1.0, "impact": 2.0, "carry": float("inf")}
            ),
            "finite or None",
        ),
        (
            lambda: MorrisEvaluation(
                "evaluated_no_impact",
                {"state": 1.0, "impact": 2.0, "carry": None},
            ),
            "no-impact",
        ),
        (
            lambda: MorrisEvaluation(
                "numerical_failure",
                {"state": None, "impact": None, "carry": 4.0},
            ),
            "failure outputs",
        ),
    ],
)
def test_malformed_evaluations_abort_without_fabrication(
    evaluation_factory: Callable[[], MorrisEvaluation], message: str
) -> None:
    with pytest.raises(ContractViolationError, match=message):
        evaluate_morris_design(
            _design(1), _outputs(), lambda _sample: evaluation_factory()
        )


def test_wrong_evaluator_return_type_aborts() -> None:
    with pytest.raises(ContractViolationError, match="MorrisEvaluation"):
        evaluate_morris_design(  # type: ignore[arg-type]
            _design(1), _outputs(), lambda _sample: {"state": 1.0}
        )


def test_unexpected_evaluator_exception_propagates() -> None:
    def bug(_sample: MorrisSample) -> MorrisEvaluation:
        raise RuntimeError("programming defect")

    with pytest.raises(RuntimeError, match="programming defect"):
        evaluate_morris_design(
            _design(1), _outputs(), bug, MorrisExecutionOptions(n_workers=2)
        )


def test_progress_emits_completed_prefix_every_eight_samples_plus_final() -> None:
    reports: list[object] = []

    evaluate_morris_design(
        _design(3),
        _outputs(),
        _hit,
        MorrisExecutionOptions(n_workers=4, progress_cb=reports.append),
    )

    assert [report.iteration for report in reports] == [8, 9]  # type: ignore[attr-defined]
    assert [report.cost for report in reports] == [0.0, 0.0]  # type: ignore[attr-defined]
    assert all(report.elapsed_s >= 0.0 for report in reports)  # type: ignore[attr-defined]


def test_progress_failure_count_is_cumulative_and_canonical() -> None:
    reports: list[object] = []

    def alternating(sample: MorrisSample) -> MorrisEvaluation:
        if sample.ordinal in (1, 4):
            return MorrisEvaluation(
                "numerical_failure",
                {"state": None, "impact": None, "carry": None},
            )
        return _hit(sample)

    evaluate_morris_design(
        _design(3),
        _outputs(),
        alternating,
        MorrisExecutionOptions(n_workers=3, progress_cb=reports.append),
    )

    assert [report.iteration for report in reports] == [8, 9]  # type: ignore[attr-defined]
    assert [report.cost for report in reports] == [2.0, 2.0]  # type: ignore[attr-defined]


def test_pre_set_cancel_event_raises_without_evaluation() -> None:
    event = threading.Event()
    event.set()
    calls = 0

    def evaluator(sample: MorrisSample) -> MorrisEvaluation:
        nonlocal calls
        calls += 1
        return _hit(sample)

    with pytest.raises(CancelledError, match="before start"):
        evaluate_morris_design(
            _design(1),
            _outputs(),
            evaluator,
            MorrisExecutionOptions(cancel_event=event),
        )
    assert calls == 0


def test_mid_run_cancellation_returns_no_partial_result() -> None:
    event = threading.Event()

    def cancel_after_first_report(_report: object) -> None:
        event.set()

    with pytest.raises(CancelledError, match="cancelled"):
        evaluate_morris_design(
            _design(8),
            _outputs(),
            _hit,
            MorrisExecutionOptions(
                n_workers=3,
                progress_cb=cancel_after_first_report,
                cancel_event=event,
            ),
        )


@pytest.mark.parametrize("n_workers", [True, 0, MAX_MORRIS_WORKERS + 1, 1.5])
def test_worker_count_requires_bounded_true_integer(n_workers: object) -> None:
    with pytest.raises(ContractViolationError, match="n_workers"):
        MorrisExecutionOptions(n_workers=n_workers)  # type: ignore[arg-type]


def test_sample_count_has_named_resource_bound() -> None:
    trajectories = MAX_MORRIS_SAMPLES // 3 + 1
    oversized = _design(trajectories)

    with pytest.raises(ContractViolationError, match="MAX_MORRIS_SAMPLES"):
        evaluate_morris_design(oversized, _outputs(), _hit)


def test_sample_output_allocation_has_named_resource_bound() -> None:
    sample_count = MAX_MORRIS_SAMPLES - (MAX_MORRIS_SAMPLES % 3)
    design = _design(sample_count // 3)
    output_count = MAX_MORRIS_OBSERVATION_CELLS // sample_count + 1
    outputs = tuple(
        MorrisOutput(f"output-{index}", "m") for index in range(output_count)
    )

    with pytest.raises(ContractViolationError, match="MAX_MORRIS_OBSERVATION_CELLS"):
        evaluate_morris_design(
            design, outputs, lambda _sample: MorrisEvaluation("evaluated_hit", {})
        )


def test_inputs_require_canonical_types_and_unique_output_names() -> None:
    outputs = _outputs()
    duplicate = (outputs[0], outputs[0])
    with pytest.raises(ContractViolationError, match="MorrisDesign"):
        evaluate_morris_design(object(), outputs, _hit)  # type: ignore[arg-type]
    with pytest.raises(ContractViolationError, match="MorrisOutput"):
        evaluate_morris_design(_design(1), (object(),), _hit)  # type: ignore[arg-type]
    with pytest.raises(ContractViolationError, match="unique"):
        evaluate_morris_design(_design(1), duplicate, _hit)


def test_evaluator_must_be_callable() -> None:
    with pytest.raises(ContractViolationError, match="callable"):
        evaluate_morris_design(_design(1), _outputs(), object())  # type: ignore[arg-type]
