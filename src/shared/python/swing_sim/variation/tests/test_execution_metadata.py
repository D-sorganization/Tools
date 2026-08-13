from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.variation import NoiseSpec, VariationPlan
from shared.python.swing_sim.variation.execution_metadata import (
    LEGACY_CURRENT_REGISTRY_WARNING,
    execution_document_from_json_dict,
    execution_document_to_json_dict,
    make_execution_metadata,
    resolve_execution_metadata,
)

_BALL_SPEED = "swing_sim.flight.launch.ball_speed_mph"
_LAUNCH_ANGLE = "swing_sim.flight.launch.launch_angle_deg"
_FIXTURE = (
    Path(__file__).parents[5]
    / "rate_of_closure/web/src/model/__fixtures__/variation_execution_document_v1.json"
)


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


def test_metadata_snapshots_exact_resolved_values_units_and_dimensions() -> None:
    metadata = make_execution_metadata(_plan())

    assert metadata.schema_id == "rate-of-closure/variation-execution-metadata"
    assert metadata.schema_version == 1
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
def test_execution_document_rejects_missing_or_duplicate_snapshots(mutation: str) -> None:
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


def test_python_matches_shared_execution_document_fixture() -> None:
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))

    assert execution_document_to_json_dict(_plan()) == fixture
    assert execution_document_from_json_dict(fixture).plan == _plan()
