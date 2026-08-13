"""Lossless Morris raw-observation archive contract tests."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.variation import (
    MORRIS_OBSERVATION_SCHEMA_ID,
    MorrisEvaluation,
    MorrisFactor,
    MorrisOutput,
    analyze_morris,
    evaluate_morris_design,
    generate_morris_design,
    morris_observations_from_json_dict,
    morris_observations_to_json_dict,
)


def _archive() -> dict[str, Any]:
    factor = MorrisFactor(
        "face-source",
        "swing_sim.impact.delivery.face_angle_deg",
        -2.0,
        2.0,
        "deg",
    )
    design = generate_morris_design((factor,), trajectories=2, levels=4, seed=7)
    outputs = (
        MorrisOutput("state", "m"),
        MorrisOutput("impact", "deg", target_kind="impact"),
    )

    def evaluate(sample: object) -> MorrisEvaluation:
        ordinal = sample.ordinal  # type: ignore[attr-defined]
        if ordinal == 1:
            return MorrisEvaluation(
                "evaluated_no_impact", {"state": 2.0, "impact": None}
            )
        if ordinal == 2:
            return MorrisEvaluation(
                "numerical_failure",
                {"state": None, "impact": None},
                failure_type="ConvergenceError",
                failure_message="bounded diagnostic",
            )
        return MorrisEvaluation(
            "evaluated_hit", {"state": float(ordinal), "impact": 3.0}
        )

    observations = evaluate_morris_design(design, outputs, evaluate)
    return morris_observations_to_json_dict(
        observations,
        study_id="study-alpha",
        provenance={
            "producer": "test-suite",
            "rng_algorithm": "numpy-pcg64",
            "request_sha256": "a" * 64,
        },
    )


def test_raw_archive_round_trip_is_exact_and_recomputes_report() -> None:
    document = _archive()
    assert document["schema_id"] == MORRIS_OBSERVATION_SCHEMA_ID
    records = document["records"]
    assert [record["ordinal"] for record in records] == list(range(4))
    assert records[1]["status"] == "evaluated_no_impact"
    assert records[1]["outputs"][1]["value"] is None
    assert records[2]["failure_type"] == "ConvergenceError"
    assert records[2]["failure_message"] == "bounded diagnostic"
    assert len({record["sample_id"] for record in records}) == 4

    archive = morris_observations_from_json_dict(copy.deepcopy(document))
    restored = archive.observations
    assert archive.study_id == "study-alpha"
    assert dict(archive.provenance)["request_sha256"] == "a" * 64
    assert restored.failure_types[1, 0] == "ConvergenceError"
    original_report = analyze_morris(restored, minimum_effects=2).to_json_dict()
    round_trip_report = analyze_morris(
        morris_observations_from_json_dict(document).observations, minimum_effects=2
    ).to_json_dict()
    assert round_trip_report == original_report


def test_raw_archive_bytes_and_sample_identity_are_deterministic() -> None:
    first = _archive()
    second = _archive()
    assert json.dumps(first, ensure_ascii=False, separators=(",", ":")) == json.dumps(
        second, ensure_ascii=False, separators=(",", ":")
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(extra=True),
        lambda value: value.update(schema_version=2),
        lambda value: value.update(schema_version=True),
        lambda value: value["records"][0].update(ordinal=2),
        lambda value: value["records"][0].update(sample_id="crossed"),
        lambda value: value["records"][1]["outputs"][1].update(value=0.0),
        lambda value: value["records"][0]["outputs"][1].update(value=None),
        lambda value: value["records"][2].update(failure_type=None),
        lambda value: value["records"][2].update(
            failure_type=None, failure_message=None
        ),
        lambda value: value["design"].update(seed=8),
    ],
)
def test_raw_archive_rejects_unknown_crossed_or_fabricated_data(
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    document = _archive()
    mutation(document)
    with pytest.raises((ContractViolationError, TypeError, ValueError)):
        morris_observations_from_json_dict(document)


def test_raw_archive_owns_immutable_arrays() -> None:
    archive = morris_observations_from_json_dict(_archive())
    assert not archive.observations.values.flags.writeable
    assert not archive.observations.outcomes.flags.writeable
    assert not archive.observations.failure_types.flags.writeable
    with pytest.raises(ValueError):
        archive.observations.values[0, 0, 0] = 99.0


def test_serializer_rejects_hit_with_unavailable_downstream_value() -> None:
    document = _archive()
    archive = morris_observations_from_json_dict(document)
    observations = archive.observations
    values = observations.values.copy()
    values[0, 0, 1] = np.nan
    incomplete = type(observations)(
        observations.design,
        observations.outputs,
        values,
        observations.outcomes,
        observations.failure_types,
        observations.failure_messages,
    )
    with pytest.raises(ContractViolationError, match="every impact and shot"):
        morris_observations_to_json_dict(
            incomplete, study_id="study", provenance={"producer": "test"}
        )


def test_serializer_rejects_shared_identifiers_that_are_not_wire_safe() -> None:
    factor = MorrisFactor(
        "face\nsource",
        "swing_sim.impact.delivery.face_angle_deg",
        -1.0,
        1.0,
        "deg",
    )
    design = generate_morris_design((factor,), 2, seed=4)
    observations = evaluate_morris_design(
        design,
        (MorrisOutput("state", "m"),),
        lambda _sample: MorrisEvaluation("evaluated_hit", {"state": 1.0}),
    )
    with pytest.raises(ContractViolationError, match="bounded nonempty"):
        morris_observations_to_json_dict(
            observations, study_id="study", provenance={"producer": "test"}
        )
