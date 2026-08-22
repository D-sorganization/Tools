"""Governed measurements for bounded durable ensemble execution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rate_of_closure.variation.ensemble_scaling_evidence import (
    ScalingEvidenceError,
    parse_scaling_evidence,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _artifact() -> dict[str, object]:
    path = (
        Path(__file__).parents[2]
        / "docs"
        / "rate_of_closure"
        / "ensemble_stream_scaling.v1.json"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_checked_in_scaling_evidence_passes_declared_budgets() -> None:
    evidence = parse_scaling_evidence(_artifact())

    assert evidence.schema_id == "rate-of-closure/ensemble-stream-scaling-evidence"
    assert evidence.schema_version == 1
    assert evidence.measurement_policy == "synthetic-failure-transport-diagnostic"
    assert len(evidence.source_commit) in {40, 64}
    assert evidence.passed
    assert len(evidence.observations) >= 3
    assert {item.trial_count for item in evidence.observations} >= {128, 512}
    assert {item.trace_sample_count for item in evidence.observations} >= {51, 501}
    assert all(item.archive_bytes > 0 for item in evidence.observations)
    assert all(item.peak_resident_bytes > 0 for item in evidence.observations)
    assert all(item.trials_per_second > 0.0 for item in evidence.observations)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("schema_version",), True, "schema_version"),
        (("budgets", "max_peak_resident_bytes"), -1, "resident"),
        (("observations", 0, "elapsed_s"), float("nan"), "elapsed"),
        (("observations", 0, "archive_bytes"), True, "archive"),
        (("observations", 0, "trial_count"), 0, "trial_count"),
        (("observations", 0, "passed"), False, "observation budget"),
    ],
)
def test_scaling_evidence_rejects_invalid_or_failed_claims(
    path: tuple[str | int, ...], value: object, message: str
) -> None:
    document: object = _artifact()
    target = document
    for key in path[:-1]:
        assert isinstance(target, (dict, list))
        target = target[key]  # type: ignore[index]
    assert isinstance(target, (dict, list))
    target[path[-1]] = value  # type: ignore[index]

    with pytest.raises(ScalingEvidenceError, match=message):
        parse_scaling_evidence(document)


def test_scaling_evidence_recomputes_throughput_and_budget_status() -> None:
    document = _artifact()
    observations = document["observations"]
    assert isinstance(observations, list)
    first = observations[0]
    assert isinstance(first, dict)
    first["trials_per_second"] = float(first["trials_per_second"]) * 2.0

    with pytest.raises(ScalingEvidenceError, match="throughput"):
        parse_scaling_evidence(document)


def test_scaling_evidence_requires_independent_trial_and_trace_axes() -> None:
    document = _artifact()
    observations = document["observations"]
    assert isinstance(observations, list)
    first = observations[0]
    assert isinstance(first, dict)
    document["observations"] = [dict(first), dict(first), dict(first)]

    with pytest.raises(ScalingEvidenceError, match="independent scaling axes"):
        parse_scaling_evidence(document)
