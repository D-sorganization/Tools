"""Strict authority requests and lifecycle envelopes for durable ensembles."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from rate_of_closure.application.durable_ensemble.contracts import (
    DURABLE_ENSEMBLE_JOB_SCHEMA_ID,
    DURABLE_ENSEMBLE_REQUEST_SCHEMA_ID,
    DurableEnsembleJobEnvelope,
    durable_ensemble_request_document,
    parse_durable_ensemble_job,
    parse_durable_ensemble_request,
)
from rate_of_closure.variation import (
    durable_ensemble_evidence,
)
from shared.python.swing_sim.variation import VariationPlan

from .test_variation_durable_ensemble_evidence import _summary
from .test_variation_simulation_request import _YAW, _base_config, _spec

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _plan() -> VariationPlan:
    return VariationPlan(
        mode="swing",
        noise=(_spec(_YAW, 0.5),),
        n_runs=3,
        seed=41,
    )


def test_request_round_trip_builds_lazy_exact_source() -> None:
    document = durable_ensemble_request_document(
        "request-41", "campaign-41", _plan(), _base_config(), chunk_size=2
    )

    request = parse_durable_ensemble_request(document)
    source = request.source()

    assert document["schema_id"] == DURABLE_ENSEMBLE_REQUEST_SCHEMA_ID
    assert request.request_id == "request-41"
    assert request.archive_id == "campaign-41"
    assert request.chunk_size == 2
    assert isinstance(document["plan_sha256"], str)
    assert source.plan == _plan()
    assert not hasattr(source, "sampled_inputs")
    assert not hasattr(source, "configs")


def test_request_rejects_plan_digest_substitution() -> None:
    document = durable_ensemble_request_document(
        "request-41", "campaign-41", _plan(), _base_config(), chunk_size=2
    )
    document["plan"]["seed"] += 1

    with pytest.raises(ValueError, match="plan digest mismatch"):
        parse_durable_ensemble_request(document)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda item: item.update(directory="C:/private"), "fields"),
        (lambda item: item.update(archive_id="../escape"), "archive_id"),
        (lambda item: item.update(scope="unbounded"), "scope"),
        (lambda item: item.update(chunk_size=True), "chunk_size"),
    ],
)
def test_request_rejects_untrusted_scope_and_path_drift(
    mutation: Callable[[dict[str, object]], None], message: str
) -> None:
    document = durable_ensemble_request_document(
        "request-41", "campaign-41", _plan(), _base_config(), chunk_size=2
    )
    mutation(document)

    with pytest.raises((TypeError, ValueError), match=message):
        parse_durable_ensemble_request(document)


def test_job_envelope_carries_incremental_path_free_evidence(tmp_path: Path) -> None:
    evidence = durable_ensemble_evidence(_summary(tmp_path))
    envelope = DurableEnsembleJobEnvelope(
        "job-1",
        "request-41",
        "campaign-41",
        "running",
        3,
        5,
        False,
        evidence,
        None,
    )

    document = envelope.to_json_dict()

    assert document["schema_id"] == DURABLE_ENSEMBLE_JOB_SCHEMA_ID
    assert document["evidence"]["archive"]["analyzed_trial_count"] == 3
    assert "directory" not in str(document)
    assert parse_durable_ensemble_job(document) == envelope
