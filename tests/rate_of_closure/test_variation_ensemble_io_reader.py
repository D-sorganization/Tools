"""Strict round-trip and corruption tests for complete Rate ensembles."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from rate_of_closure.variation import _ensemble_parser as ensemble_parser
from rate_of_closure.variation import ensemble_io
from rate_of_closure.variation._ensemble_limits import (
    MAX_POINTS,
    MAX_SAMPLES,
    MAX_TRIALS,
    require_ensemble_shape_limits,
)
from rate_of_closure.variation.ensemble_io import (
    ENSEMBLE_EXPORT_SCHEMA_VERSION,
    from_json_dict,
    read_json,
    to_json_dict,
    write_json,
)
from rate_of_closure.variation.simulation_types import (
    ALL_OUTPUT_NAMES,
    APP_FRAME_ID,
    CONTACT_OUTPUT_NAMES,
    EVALUATED_HIT,
    EVALUATED_NO_IMPACT,
    NUMERICAL_FAILURE,
    SimulationEnsembleResult,
    SimulationTrialOutcome,
)
from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.variation import (
    NoiseSpec,
    PerturbationGroup,
    VariationPlan,
)
from shared.python.swing_sim.variation.engine import VariationDataset
from shared.python.swing_sim.variation.ensemble_types import EnsemblePositionTraces
from shared.python.swing_sim.variation.registry import CATEGORY_DELIVERY

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_FACE = f"{CATEGORY_DELIVERY}.face_angle_deg"
_PATH = f"{CATEGORY_DELIVERY}.club_path_deg"


def _values(start: float) -> dict[str, float]:
    return {name: start + index for index, name in enumerate(ALL_OUTPUT_NAMES)}


def _result() -> SimulationEnsembleResult:
    plan = VariationPlan(
        mode="delivery",
        noise=(
            NoiseSpec(
                _FACE,
                scale=1.0,
                spec_id="face-noise",
                time_window_s=(0.0, 0.01),
                point_ids=("swing.clubhead.reference",),
            ),
            NoiseSpec(_PATH, scale=2.0, spec_id="path-noise"),
        ),
        n_runs=3,
        seed=17,
        groups=(
            PerturbationGroup(
                group_id="delivery-correlation",
                spec_ids=("face-noise", "path-noise"),
                matrix=((1.0, 0.25), (0.25, 1.0)),
            ),
        ),
    )
    hit_values: dict[str, float | None] = _values(1.0)
    miss_values: dict[str, float | None] = {
        name: (10.0 + index if name in CONTACT_OUTPUT_NAMES else None)
        for index, name in enumerate(ALL_OUTPUT_NAMES)
    }
    failure_values: dict[str, float | None] = {name: None for name in ALL_OUTPUT_NAMES}
    outcomes = (
        SimulationTrialOutcome(0, EVALUATED_HIT, hit_values),
        SimulationTrialOutcome(1, EVALUATED_NO_IMPACT, miss_values),
        SimulationTrialOutcome(
            2,
            NUMERICAL_FAILURE,
            failure_values,
            failure_type="FloatingPointError",
            failure_message="non-finite state",
        ),
    )
    outputs = np.array(
        [
            [hit_values[name] for name in ALL_OUTPUT_NAMES],
            [
                np.nan if miss_values[name] is None else miss_values[name]
                for name in ALL_OUTPUT_NAMES
            ],
            [np.nan for _name in ALL_OUTPUT_NAMES],
        ],
        dtype=float,
    )
    variation = VariationDataset(
        plan=plan,
        input_names=(_FACE, _PATH),
        inputs=np.array([[-1.0, -2.0], [0.0, 0.0], [1.0, 2.0]]),
        output_names=ALL_OUTPUT_NAMES,
        outputs=outputs,
        success=np.array([True, True, False]),
        elapsed_s=0.25,
    )
    positions = np.array(
        [
            [[[0.0, 1.0, 2.0]], [[1.0, 2.0, 3.0]]],
            [[[2.0, 3.0, 4.0]], [[3.0, 4.0, 5.0]]],
            [[[np.nan, np.nan, np.nan]], [[np.nan, np.nan, np.nan]]],
        ]
    )
    traces = EnsemblePositionTraces(
        variation=variation,
        sample_times_s=np.array([0.0, 0.01]),
        coordinate_frame=APP_FRAME_ID,
        point_ids=("swing.clubhead.reference",),
        positions_m=positions,
        sample_valid=np.array([[True, True], [True, True], [False, False]]),
        impact_sample_indices=np.array([1, -1, -1]),
    )
    return SimulationEnsembleResult(outcomes, variation, traces)


def test_reader_round_trips_writer_and_owns_immutable_arrays(tmp_path) -> None:
    source = _result()
    document = to_json_dict(source)

    parsed = from_json_dict(document)

    assert to_json_dict(parsed) == document
    assert parsed.variation.plan.noise[0].spec_id == "face-noise"
    assert parsed.variation.plan.noise[0].time_window_s == (0.0, 0.01)
    assert parsed.variation.plan.noise[0].point_ids == ("swing.clubhead.reference",)
    assert parsed.variation.plan.groups[0].group_id == "delivery-correlation"
    assert parsed.outcomes[1].status is EVALUATED_NO_IMPACT
    assert parsed.outcomes[2].failure_type == "FloatingPointError"
    for array in (
        parsed.variation.inputs,
        parsed.variation.outputs,
        parsed.variation.success,
        parsed.traces.sample_times_s,
        parsed.traces.positions_m,
        parsed.traces.sample_valid,
        parsed.traces.impact_sample_indices,
    ):
        assert array.flags.owndata
        assert not array.flags.writeable

    path = tmp_path / "ensemble.json"
    write_json(source, path)
    assert to_json_dict(read_json(path)) == document


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data.update(schema_version=True), "schema_version"),
        (lambda data: data.update(schema_version=2), "unsupported schema_version"),
        (lambda data: data.update(position_unit="ft"), "position_unit"),
        (lambda data: data.update(time_unit="ms"), "time_unit"),
        (
            lambda data: data.update(coordinate_frame="app_frame:z_up"),
            "coordinate_frame",
        ),
        (lambda data: data.update(extra="ambient"), "root fields"),
    ],
)
def test_reader_rejects_schema_drift(mutation, message: str) -> None:
    document = to_json_dict(_result())
    mutation(document)

    with pytest.raises(ContractViolationError, match=message):
        from_json_dict(document)


def test_reader_rejects_crossed_outcomes_variation_and_trace_status() -> None:
    document = to_json_dict(_result())
    crossed = copy.deepcopy(document)
    crossed["outcomes"][0]["values"]["carry_m"] += 1.0
    with pytest.raises(ContractViolationError, match="variation outputs"):
        from_json_dict(crossed)

    crossed = copy.deepcopy(document)
    crossed["variation"]["success"][1] = False
    with pytest.raises(ContractViolationError, match="variation success"):
        from_json_dict(crossed)

    crossed = copy.deepcopy(document)
    crossed["impact_sample_indices"][0] = -1
    with pytest.raises(ContractViolationError, match="impact marker"):
        from_json_dict(crossed)

    crossed = copy.deepcopy(document)
    crossed["impact_sample_indices"][0] = 0
    with pytest.raises(ContractViolationError, match="impact-time provenance"):
        from_json_dict(crossed)


def test_reader_rejects_corrupt_availability_and_identifiers() -> None:
    document = to_json_dict(_result())
    corrupt = copy.deepcopy(document)
    del corrupt["outcomes"][0]["values"]["carry_m"]
    with pytest.raises(ContractViolationError, match="scalar output fields"):
        from_json_dict(corrupt)

    corrupt = copy.deepcopy(document)
    corrupt["outcomes"][1]["values"]["carry_m"] = 0.0
    with pytest.raises(ContractViolationError, match="impact and shot outputs"):
        from_json_dict(corrupt)

    corrupt = copy.deepcopy(document)
    corrupt["point_ids"] = ["swing.clubhead.reference", "swing.clubhead.reference"]
    with pytest.raises(ContractViolationError, match="unique"):
        from_json_dict(corrupt)

    corrupt = copy.deepcopy(document)
    corrupt["outcomes"][1]["trial_index"] = 0
    with pytest.raises(ContractViolationError, match="canonical trial order"):
        from_json_dict(corrupt)


def test_file_reader_rejects_duplicate_truncated_and_oversized_json(tmp_path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(ContractViolationError, match="duplicate JSON field"):
        read_json(duplicate)

    truncated = tmp_path / "truncated.json"
    truncated.write_text('{"schema_version":', encoding="utf-8")
    with pytest.raises(ContractViolationError, match="valid JSON"):
        read_json(truncated)

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * (16_000_001))
    with pytest.raises(ContractViolationError, match="byte limit"):
        read_json(oversized)


def test_version_one_is_the_only_supported_outer_schema() -> None:
    document = to_json_dict(_result())
    assert document["schema_version"] == ENSEMBLE_EXPORT_SCHEMA_VERSION == 1

    document["schema_version"] = 0
    with pytest.raises(ContractViolationError, match="unsupported schema_version"):
        from_json_dict(document)


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (("variation", "plan", "n_runs"), True, "n_runs must be an integer"),
        (("variation", "plan", "seed"), 1.0, "seed must be an integer"),
        (("variation", "plan", "noise", 0, "scale"), "1.0", "noise scale"),
        (("variation", "success", 0), 1, "success must contain booleans"),
        (("impact_sample_indices", 0), True, "must be an integer"),
    ],
)
def test_reader_rejects_coercible_noncanonical_scalar_types(
    path: tuple[object, ...], replacement: object, message: str
) -> None:
    document: object = to_json_dict(_result())
    target = document
    for segment in path[:-1]:
        target = target[segment]
    target[path[-1]] = replacement

    with pytest.raises(ContractViolationError, match=message):
        from_json_dict(document)


def test_reader_rejects_excessive_decoded_depth_before_materialization() -> None:
    document = to_json_dict(_result())
    nested: object = 0
    for _ in range(40):
        nested = [nested]
    document["extra"] = nested

    with pytest.raises(ContractViolationError, match="nesting depth"):
        from_json_dict(document)


def test_reader_does_not_alias_mutable_source_document() -> None:
    document = to_json_dict(_result())
    parsed = from_json_dict(document)

    document["variation"]["inputs"][0][0] = 999.0
    document["positions_m"][0][0][0][0] = 999.0
    document["outcomes"][0]["values"]["carry_m"] = 999.0

    assert parsed.variation.inputs[0, 0] == -1.0
    assert parsed.traces.positions_m[0, 0, 0, 0] == 0.0
    assert parsed.outcomes[0].value("carry_m") != 999.0


def test_typed_result_enforces_the_same_point_limit_as_the_reader() -> None:
    source = _result()
    point_ids = tuple(
        f"swing.synthetic.point-{index}"
        for index in range(ensemble_parser.MAX_POINTS + 1)
    )
    positions = np.repeat(source.traces.positions_m, len(point_ids), axis=2)
    traces = EnsemblePositionTraces(
        variation=source.variation,
        sample_times_s=source.traces.sample_times_s,
        coordinate_frame=source.traces.coordinate_frame,
        point_ids=point_ids,
        positions_m=positions,
        sample_valid=source.traces.sample_valid,
        impact_sample_indices=source.traces.impact_sample_indices,
    )

    with pytest.raises(ContractViolationError, match="point limit"):
        SimulationEnsembleResult(source.outcomes, source.variation, traces)


def test_typed_result_rejects_crossed_scalar_authority() -> None:
    source = _result()
    outputs = source.variation.outputs.copy()
    outputs[0, ALL_OUTPUT_NAMES.index("carry_m")] += 1.0
    variation = VariationDataset(
        plan=source.variation.plan,
        input_names=source.variation.input_names,
        inputs=source.variation.inputs,
        output_names=source.variation.output_names,
        outputs=outputs,
        success=source.variation.success,
        elapsed_s=source.variation.elapsed_s,
    )
    traces = EnsemblePositionTraces(
        variation=variation,
        sample_times_s=source.traces.sample_times_s,
        coordinate_frame=source.traces.coordinate_frame,
        point_ids=source.traces.point_ids,
        positions_m=source.traces.positions_m,
        sample_valid=source.traces.sample_valid,
        impact_sample_indices=source.traces.impact_sample_indices,
    )

    with pytest.raises(ContractViolationError, match="variation outputs"):
        SimulationEnsembleResult(source.outcomes, variation, traces)


@pytest.mark.parametrize("elapsed_s", [-1.0, np.nan, np.inf])
def test_typed_result_rejects_invalid_elapsed_time(elapsed_s: float) -> None:
    source = _result()
    variation = VariationDataset(
        plan=source.variation.plan,
        input_names=source.variation.input_names,
        inputs=source.variation.inputs,
        output_names=source.variation.output_names,
        outputs=source.variation.outputs,
        success=source.variation.success,
        elapsed_s=elapsed_s,
    )
    traces = EnsemblePositionTraces(
        variation=variation,
        sample_times_s=source.traces.sample_times_s,
        coordinate_frame=source.traces.coordinate_frame,
        point_ids=source.traces.point_ids,
        positions_m=source.traces.positions_m,
        sample_valid=source.traces.sample_valid,
        impact_sample_indices=source.traces.impact_sample_indices,
    )
    with pytest.raises(ContractViolationError, match="elapsed_s"):
        SimulationEnsembleResult(source.outcomes, variation, traces)


def test_typed_result_rejects_nonfinite_sampled_inputs() -> None:
    source = _result()
    inputs = source.variation.inputs.copy()
    inputs[0, 0] = np.nan
    variation = VariationDataset(
        plan=source.variation.plan,
        input_names=source.variation.input_names,
        inputs=inputs,
        output_names=source.variation.output_names,
        outputs=source.variation.outputs,
        success=source.variation.success,
        elapsed_s=source.variation.elapsed_s,
    )
    traces = EnsemblePositionTraces(
        variation=variation,
        sample_times_s=source.traces.sample_times_s,
        coordinate_frame=source.traces.coordinate_frame,
        point_ids=source.traces.point_ids,
        positions_m=source.traces.positions_m,
        sample_valid=source.traces.sample_valid,
        impact_sample_indices=source.traces.impact_sample_indices,
    )

    with pytest.raises(ContractViolationError, match="sampled inputs must be finite"):
        SimulationEnsembleResult(source.outcomes, variation, traces)


def test_sample_limit_is_checked_before_numeric_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = to_json_dict(_result())
    document["sample_times_s"] = [0.0] * (ensemble_parser.MAX_SAMPLES + 1)

    def unexpected_materialization(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("sample vector materialized before its bound")

    monkeypatch.setattr(ensemble_parser, "number_vector", unexpected_materialization)
    with pytest.raises(ContractViolationError, match="sample limit"):
        from_json_dict(document)


@pytest.mark.parametrize(
    "text",
    [
        "[" * 2_000 + "0" + "]" * 2_000,
        '{"value":' + "9" * 5_000 + "}",
    ],
)
def test_text_reader_normalizes_decoder_resource_errors(text: str) -> None:
    with pytest.raises(ContractViolationError, match="valid JSON|nesting depth"):
        ensemble_io.loads(text)


def test_file_writer_fails_before_creating_an_unreadable_oversized_file(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "ensemble.json"
    monkeypatch.setattr(ensemble_io, "MAX_ENSEMBLE_JSON_BYTES", 32)

    with pytest.raises(ContractViolationError, match="byte limit"):
        write_json(_result(), path)

    assert not path.exists()


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((MAX_TRIALS + 1, 1, 1), "trial limit"),
        ((1, MAX_SAMPLES + 1, 1), "sample limit"),
        ((1, 1, MAX_POINTS + 1), "point limit"),
        ((20, 1_000, 100), "position cell limit"),
    ],
)
def test_shared_shape_limits_reject_each_resource_axis(
    shape: tuple[int, int, int], message: str
) -> None:
    with pytest.raises(ContractViolationError, match=message):
        require_ensemble_shape_limits(*shape)


def test_position_axes_are_preflighted_before_tensor_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = to_json_dict(_result())
    document["positions_m"][0][0] = []

    def unexpected_allocation(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("position tensor allocated before axis validation")

    monkeypatch.setattr(ensemble_parser.np, "full", unexpected_allocation)
    with pytest.raises(ContractViolationError, match="point axis"):
        from_json_dict(document)


def test_file_writer_rejects_nonstandard_json_before_creating_file(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "ensemble.json"
    monkeypatch.setattr(
        ensemble_io,
        "to_json_dict",
        lambda _result: {"nonfinite": float("nan")},
    )

    with pytest.raises(ContractViolationError, match="strict finite JSON"):
        write_json(_result(), path)

    assert not path.exists()
