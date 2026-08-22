from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.variation import NoiseSpec, VariationPlan
from shared.python.swing_sim.variation.execution_metadata import (
    LEGACY_CURRENT_REGISTRY_WARNING,
    LEGACY_EXECUTION_DOCUMENT_MIGRATION_ERROR,
    execution_document_from_json_dict,
    execution_document_to_json_dict,
    make_execution_metadata,
    resolve_execution_metadata,
)

_BALL_SPEED = "swing_sim.flight.launch.ball_speed_mph"
_LAUNCH_ANGLE = "swing_sim.flight.launch.launch_angle_deg"
_LAUNCH_AZIMUTH = "swing_sim.flight.launch.launch_azimuth_deg"
_MAX_SAFE_INTEGER = 9_007_199_254_740_991
_V1_FIXTURE = (
    Path(__file__).parents[5]
    / "rate_of_closure/web/src/model/__fixtures__/variation_execution_document_v1.json"
)
_V1_EDGE_FIXTURE = (
    Path(__file__).parents[5]
    / "rate_of_closure/web/src/model/__fixtures__"
    / "variation_execution_document_edge_floats_v1.json"
)
_PYTHON_V2_FIXTURE = _V1_FIXTURE.with_name(
    "variation_execution_document_python_v2.json"
)
_REACT_V2_FIXTURE = _V1_FIXTURE.with_name("variation_execution_document_react_v2.json")


def _plan(seed: int = 17) -> VariationPlan:
    return VariationPlan(
        mode="launch",
        base_variables={_BALL_SPEED: 154.25},
        noise=(
            NoiseSpec(
                _LAUNCH_ANGLE,
                distribution="normal",
                scale=0.75,
                spec_id="launch-angle",
            ),
        ),
        n_runs=8,
        seed=seed,
        flight_model="waterloo_penner",
    )


def _edge_plan() -> VariationPlan:
    return VariationPlan(
        mode="launch",
        base_variables={
            _BALL_SPEED: 154.00000000000003,
            _LAUNCH_AZIMUTH: -0.0,
            "swing_sim.flight.launch.spin_axis_deg": 1.0000000000000002,
        },
        noise=(
            NoiseSpec(
                _LAUNCH_ANGLE,
                distribution="normal",
                scale=0.5000000000000001,
                spec_id="edge-angle",
            ),
        ),
        n_runs=8,
        seed=_MAX_SAFE_INTEGER,
        flight_model="waterloo_penner",
    )


def test_metadata_snapshots_exact_resolved_values_units_and_dimensions() -> None:
    metadata = make_execution_metadata(_plan())

    assert metadata.schema_id == "rate-of-closure/variation-execution-metadata"
    assert metadata.schema_version == 2
    assert metadata.rng_identity.algorithm_id == "numpy-pcg64-canonical-rowwise-psd"
    assert metadata.rng_identity.algorithm_version == 2
    assert metadata.rng_identity.stream_derivation_id == (
        "numpy-seedsequence-safe-seed-crc32-utf8-spec-id"
    )
    assert metadata.implementation_identity.runtime_id == "rate-of-closure/python"
    assert metadata.implementation_identity.solver_id == (
        "python-configured-simulation+scipy-rk45-flight"
    )
    assert metadata.registry_schema_version == 1
    assert len(metadata.plan_sha256) == len(metadata.registry_sha256) == 64
    snapshots = {item.variable_key: item for item in metadata.resolved_variables}
    assert snapshots[_BALL_SPEED].value == 154.25
    assert snapshots[_BALL_SPEED].unit == "mph"
    assert snapshots[_BALL_SPEED].dimension == "speed"
    assert snapshots[_LAUNCH_ANGLE].value == 12.0
    assert snapshots[_LAUNCH_ANGLE].unit == "deg"
    assert snapshots[_LAUNCH_ANGLE].dimension == "angle"
    with pytest.raises(AttributeError):
        metadata.resolved_variables = ()  # type: ignore[misc]


def test_execution_document_round_trips_exact_canonical_plan_and_metadata() -> None:
    document = execution_document_to_json_dict(_plan())

    decoded = execution_document_from_json_dict(json.loads(json.dumps(document)))

    assert decoded.plan == _plan()
    assert decoded.metadata == make_execution_metadata(_plan())
    assert decoded.warning is None
    assert "execution_metadata" not in document["plan"]


def test_signed_zero_is_normalized_in_document_snapshot_and_digest() -> None:
    negative = _plan()
    negative = VariationPlan(
        mode=negative.mode,
        base_variables={**negative.base_variables, _LAUNCH_AZIMUTH: -0.0},
        noise=negative.noise,
        n_runs=negative.n_runs,
        seed=negative.seed,
        flight_model=negative.flight_model,
    )
    positive = VariationPlan(
        mode=negative.mode,
        base_variables={**negative.base_variables, _LAUNCH_AZIMUTH: 0.0},
        noise=negative.noise,
        n_runs=negative.n_runs,
        seed=negative.seed,
        flight_model=negative.flight_model,
    )

    negative_document = execution_document_to_json_dict(negative)
    snapshot = next(
        item
        for item in negative_document["metadata"]["resolved_variables"]
        if item["variable_key"] == _LAUNCH_AZIMUTH
    )
    assert (
        math.copysign(1.0, negative_document["plan"]["base_variables"][_LAUNCH_AZIMUTH])
        == 1.0
    )
    assert (
        math.copysign(1.0, negative.to_json_dict()["base_variables"][_LAUNCH_AZIMUTH])
        == 1.0
    )
    assert math.copysign(1.0, snapshot["value"]) == 1.0
    assert (
        make_execution_metadata(negative).plan_sha256
        == make_execution_metadata(positive).plan_sha256
    )


@pytest.mark.parametrize("field", ["seed", "n_runs"])
def test_plan_rejects_integers_above_shared_safe_boundary(field: str) -> None:
    kwargs = {field: _MAX_SAFE_INTEGER + 1}

    with pytest.raises(ContractViolationError, match="safe integer"):
        VariationPlan(mode="launch", noise=_plan().noise, **kwargs)


def test_max_safe_seed_remains_distinct_from_preceding_seed() -> None:
    maximum = _plan(seed=_MAX_SAFE_INTEGER)
    preceding = _plan(seed=_MAX_SAFE_INTEGER - 1)
    maximum_runs = VariationPlan(
        mode="launch", noise=_plan().noise, n_runs=_MAX_SAFE_INTEGER
    )

    assert maximum.seed == _MAX_SAFE_INTEGER
    assert maximum_runs.n_runs == _MAX_SAFE_INTEGER
    assert (
        make_execution_metadata(maximum).plan_sha256
        != make_execution_metadata(preceding).plan_sha256
    )


@pytest.mark.parametrize("unsafe_seed", [9_007_199_254_740_992, 9_007_199_254_740_993])
def test_colliding_binary64_seed_candidates_both_fail_closed(unsafe_seed: int) -> None:
    with pytest.raises(ContractViolationError, match="safe integer"):
        _plan(seed=unsafe_seed)


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (("plan", "seed"), 18, "plan digest"),
        (("metadata", "flight_model"), "nathan", "flight_model"),
        (
            ("metadata", "resolved_variables", 0, "value"),
            999.0,
            "resolved variable snapshot",
        ),
        (
            ("metadata", "resolved_variables", 0, "unit"),
            "m/s",
            "resolved variable snapshot",
        ),
        (
            ("metadata", "resolved_variables", 0, "dimension"),
            "length",
            "resolved variable snapshot",
        ),
        (("metadata", "registry_sha256"), "0" * 64, "registry digest"),
        (("metadata", "rng_identity", "algorithm_id"), "other", "RNG identity"),
        (
            ("metadata", "implementation_identity", "solver_id"),
            "other",
            "implementation identity",
        ),
    ],
)
def test_execution_document_rejects_plan_registry_and_unit_drift(
    path: tuple[object, ...], replacement: object, message: str
) -> None:
    document = copy.deepcopy(execution_document_to_json_dict(_plan()))
    target: object = document
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = replacement  # type: ignore[index]

    with pytest.raises(ContractViolationError, match=message):
        execution_document_from_json_dict(document)


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_execution_document_rejects_missing_or_duplicate_snapshots(
    mutation: str,
) -> None:
    document = copy.deepcopy(execution_document_to_json_dict(_plan()))
    snapshots = document["metadata"]["resolved_variables"]
    if mutation == "missing":
        snapshots.pop()
    else:
        snapshots.append(copy.deepcopy(snapshots[0]))

    with pytest.raises(ContractViolationError, match="resolved variable snapshot"):
        execution_document_from_json_dict(document)


@pytest.mark.parametrize("target", ["document", "metadata"])
def test_execution_document_rejects_unknown_fields(target: str) -> None:
    document = copy.deepcopy(execution_document_to_json_dict(_plan()))
    record = document if target == "document" else document["metadata"]
    record["unexpected"] = True

    with pytest.raises(ContractViolationError, match="fields mismatch"):
        execution_document_from_json_dict(document)


def test_supplied_metadata_rejects_cross_plan_before_execution() -> None:
    metadata = make_execution_metadata(_plan(seed=17))

    with pytest.raises(ContractViolationError, match="plan digest"):
        resolve_execution_metadata(_plan(seed=18), metadata)


def test_legacy_plan_resolution_is_explicit_and_warns() -> None:
    resolution = resolve_execution_metadata(_plan(), None)

    assert resolution.metadata == make_execution_metadata(_plan())
    assert resolution.warning == LEGACY_CURRENT_REGISTRY_WARNING


@pytest.mark.parametrize("path", [_V1_FIXTURE, _V1_EDGE_FIXTURE])
def test_v1_execution_documents_fail_with_explicit_migration(path: Path) -> None:
    fixture = json.loads(path.read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="Historical replay remains unproven"):
        execution_document_from_json_dict(fixture)
    assert "@1" in LEGACY_EXECUTION_DOCUMENT_MIGRATION_ERROR


def test_same_python_runtime_replay_pins_metadata_and_samples() -> None:
    from shared.python.swing_sim.variation.engine import sample_inputs

    first = sample_inputs(_plan())
    second = sample_inputs(_plan())
    assert make_execution_metadata(_plan()) == make_execution_metadata(_plan())
    assert (first == second).all()


def test_python_v2_golden_and_cross_runtime_rejection() -> None:
    python_document = json.loads(_PYTHON_V2_FIXTURE.read_text(encoding="utf-8"))
    react_document = json.loads(_REACT_V2_FIXTURE.read_text(encoding="utf-8"))

    assert execution_document_to_json_dict(_plan()) == python_document
    assert (
        python_document["metadata"]["plan_sha256"]
        == react_document["metadata"]["plan_sha256"]
    )
    assert (
        python_document["metadata"]["registry_sha256"]
        == react_document["metadata"]["registry_sha256"]
    )
    assert execution_document_from_json_dict(python_document).plan == _plan()
    with pytest.raises(ContractViolationError, match="RNG identity"):
        execution_document_from_json_dict(react_document)
