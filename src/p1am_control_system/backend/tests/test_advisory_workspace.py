"""Contracts for the synthetic, non-authoritative advisory workspace."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from advisory_workspace import (
    AdvisoryDisposition,
    AdvisoryRequest,
    AdvisoryService,
    ConstraintEnvelope,
    DispositionDecision,
)
from identity import Principal, Role

NOW = datetime(2026, 8, 3, 21, 0, tzinfo=UTC)


def _request() -> AdvisoryRequest:
    return AdvisoryRequest(
        dataset_id="SYNTHETIC.DATASET.RUN-042",
        observed_throughput=62.0,
        observed_energy=47.0,
        requested_throughput=68.0,
    )


def test_evaluation_is_reproducible_and_carries_complete_evidence() -> None:
    service = AdvisoryService(now=lambda: NOW)

    first = service.evaluate(_request())
    replay = service.evaluate(_request())

    assert replay == first
    assert first.model.model_id == "SYNTHETIC.MODEL.ADVISORY"
    assert first.model.version == "1.0.0"
    assert len(first.model.artifact_sha256) == 64
    assert first.data.dataset_id == "SYNTHETIC.DATASET.RUN-042"
    assert len(first.data.content_sha256) == 64
    assert (
        first.constraints.minimum
        <= first.recommended_setpoint
        <= first.constraints.maximum
    )
    assert first.confidence.lower <= first.confidence.estimate <= first.confidence.upper
    assert first.replay.verified is True
    assert len(first.replay.input_sha256) == 64
    assert len(first.replay.result_sha256) == 64
    assert first.authoritative_write_available is False
    assert first.data_classification == "synthetic"
    assert first.not_for_live_control is True


def test_constraints_and_confidence_reject_invalid_ranges() -> None:
    with pytest.raises(ValueError, match="minimum"):
        ConstraintEnvelope(minimum=80.0, maximum=70.0, unit="synthetic unit")

    with pytest.raises(ValueError, match="finite"):
        AdvisoryRequest(
            dataset_id="SYNTHETIC.DATASET.INVALID",
            observed_throughput=float("nan"),
            observed_energy=1.0,
            requested_throughput=2.0,
        )


def test_operator_disposition_is_attributable_and_cannot_apply_control() -> None:
    service = AdvisoryService(now=lambda: NOW)
    result = service.evaluate(_request())
    principal = Principal("operator.one", "Operator One", Role.OPERATOR)

    disposition = service.record_disposition(
        result.advisory_id,
        AdvisoryDisposition(
            decision=DispositionDecision.DEFERRED,
            reason="Review with the next synthetic operating scenario",
        ),
        principal,
    )

    assert disposition.actor == "operator.one"
    assert disposition.advisory_id == result.advisory_id
    assert disposition.applied_to_control is False
    assert service.dispositions(result.advisory_id) == (disposition,)
    assert service.result(result.advisory_id) == result

    with pytest.raises(ValueError, match="reason"):
        AdvisoryDisposition(
            decision=DispositionDecision.REJECTED,
            reason=" ",
        )
