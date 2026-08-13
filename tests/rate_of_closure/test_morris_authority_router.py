"""Bounded mountable FastAPI router and in-memory job registry tests."""

from __future__ import annotations

import json
import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rate_of_closure.application.morris import router as morris_router
from rate_of_closure.application.morris.contracts import parse_morris_request
from rate_of_closure.application.morris.router import (
    MorrisJobRegistry,
    MorrisRegistryOptions,
    create_morris_router,
)
from rate_of_closure.application.morris.service import (
    MorrisServiceResult,
    RateMorrisService,
)
from rate_of_closure.variation.simulation_types import ALL_OUTPUT_NAMES
from shared.python.swing_sim.variation import MorrisEvaluation

from .test_morris_authority_contracts import request_document

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


class BlockingService:
    """A controllable service for lifecycle and cancellation tests."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def execute(
        self, request: object, cancel: threading.Event, progress: object
    ) -> dict[str, object]:
        self.started.set()
        self.release.wait(timeout=3)
        if cancel.is_set():
            from shared.python.swing_sim.variation import CancelledError

            raise CancelledError("cancelled")
        progress(4, 4)  # type: ignore[operator]
        return {
            "schema_id": "swing-sim/morris-global-sensitivity-report",
            "schema_version": 1,
        }


def _client(
    service: object, **registry_options: object
) -> tuple[TestClient, MorrisJobRegistry]:
    registry = MorrisJobRegistry(
        service=service,
        options=MorrisRegistryOptions(**registry_options),  # type: ignore[arg-type]
    )
    app = FastAPI()
    app.include_router(create_morris_router(registry), prefix="/authority")
    return TestClient(app), registry


def test_router_create_status_and_acknowledged_cancel() -> None:
    service = BlockingService()
    client, _registry = _client(service)
    created = client.post("/authority/morris/jobs", json=request_document())
    assert created.status_code == 202
    job_id = created.json()["job_id"]
    assert service.started.wait(timeout=1)

    cancelling = client.delete(f"/authority/morris/jobs/{job_id}")
    assert cancelling.status_code == 202
    assert cancelling.json()["status"] == "running"
    assert cancelling.json()["cancel_requested"] is True
    service.release.set()

    terminal = _await_terminal(client, job_id)
    assert terminal["status"] == "cancelled"
    assert terminal["report"] is None
    assert client.delete(f"/authority/morris/jobs/{job_id}").json() == terminal


def test_router_rejects_duplicate_nonfinite_and_oversized_raw_json() -> None:
    client, _registry = _client(BlockingService(), max_body_bytes=1_500)
    duplicate = json.dumps(request_document()).replace(
        '"seed": 17', '"seed": 17, "seed": 18'
    )
    headers = {"Content-Type": "application/json"}
    assert (
        client.post(
            "/authority/morris/jobs", content=duplicate, headers=headers
        ).status_code
        == 400
    )
    nonfinite = json.dumps(request_document()).replace('"seed": 17', '"seed": NaN')
    assert (
        client.post(
            "/authority/morris/jobs", content=nonfinite, headers=headers
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/authority/morris/jobs", content=b"{" + b" " * 2_000, headers=headers
        ).status_code
        == 413
    )


def test_registry_bounds_active_jobs_and_sanitizes_failures() -> None:
    service = BlockingService()
    client, _registry = _client(service, max_active_jobs=1)
    first = client.post("/authority/morris/jobs", json=request_document())
    assert first.status_code == 202
    assert (
        client.post(
            "/authority/morris/jobs",
            json={**request_document(), "request_id": "second"},
        ).status_code
        == 429
    )
    service.release.set()


def test_registry_expires_terminal_jobs_and_sanitizes_programming_failures() -> None:
    failed = threading.Event()

    class BrokenService:
        def execute(
            self, _request: object, _cancel: object, _progress: object
        ) -> object:
            failed.set()
            raise TypeError("C:\\private\\source.py must not cross the wire")

    now = [10.0]
    options = MorrisRegistryOptions(terminal_ttl_s=1.0)
    registry = MorrisJobRegistry(BrokenService(), options, lambda: now[0])
    request = parse_morris_request(request_document())
    job_id = registry.create(request).job_id
    assert failed.wait(timeout=1)
    for _index in range(100):
        envelope = registry.status(job_id)
        if envelope.status == "failed":
            break
        threading.Event().wait(0.001)
    assert envelope.error == {
        "code": "execution_failed",
        "message": "Morris execution failed",
    }
    assert "private" not in str(envelope.to_json_dict())
    now[0] = 12.0
    with pytest.raises(KeyError):
        registry.status(job_id)
    registry.close()


def test_registry_retains_raw_authority_separately_from_report_envelope() -> None:
    def evaluate(sample: object) -> MorrisEvaluation:
        values = {name: float(sample.ordinal) for name in ALL_OUTPUT_NAMES}
        return MorrisEvaluation("evaluated_hit", values)

    service = RateMorrisService(evaluator_factory=lambda _design, _config: evaluate)
    registry = MorrisJobRegistry(service)
    request = parse_morris_request(request_document())
    job_id = registry.create(request).job_id
    for _index in range(100):
        if registry.status(job_id).status == "completed":
            break
        threading.Event().wait(0.001)
    envelope = registry.status(job_id)
    assert envelope.status == "completed"
    assert "observations" not in envelope.to_json_dict()
    archive = registry.observations(job_id)
    assert archive.study_id == request.request_id
    assert archive.observations.outcomes.shape == (2, 2)
    registry.close()


def test_registry_evicts_raw_authority_by_weight_without_losing_reports() -> None:
    def evaluate(sample: object) -> MorrisEvaluation:
        values = {name: float(sample.ordinal) for name in ALL_OUTPUT_NAMES}
        return MorrisEvaluation("evaluated_hit", values)

    service = RateMorrisService(evaluator_factory=lambda _design, _config: evaluate)
    probe = service.execute_with_observations(
        parse_morris_request(request_document()),
        threading.Event(),
        lambda _done, _total: None,
    )
    registry = MorrisJobRegistry(
        service,
        MorrisRegistryOptions(
            max_retained_observation_cells=probe.observations.observation_cells
        ),
    )
    first = registry.create(parse_morris_request(request_document())).job_id
    for _index in range(100):
        if registry.status(first).status == "completed":
            break
        threading.Event().wait(0.001)
    second_document = {**request_document(), "request_id": "second-study"}
    second = registry.create(parse_morris_request(second_document)).job_id
    for _index in range(100):
        if registry.status(second).status == "completed":
            break
        threading.Event().wait(0.001)
    for job_id in (first, second):
        assert registry.status(job_id).report is not None
    with pytest.raises(RuntimeError, match="not available"):
        registry.observations(first)
    assert registry.observations(second).study_id == "second-study"
    assert registry.status(first).report is not None
    registry.close()


def test_registry_rejects_crossed_extended_result_identity() -> None:
    base_service = RateMorrisService(
        evaluator_factory=lambda _design, _config: (
            lambda sample: MorrisEvaluation(
                "evaluated_hit",
                {name: float(sample.ordinal) for name in ALL_OUTPUT_NAMES},
            )
        )
    )
    first_request = parse_morris_request(request_document())
    crossed = base_service.execute_with_observations(
        first_request, threading.Event(), lambda _done, _total: None
    )

    class CrossedService:
        def execute(
            self, _request: object, _cancel: object, _progress: object
        ) -> dict[str, object]:
            return crossed.report

        def execute_with_observations(
            self, _request: object, _cancel: object, _progress: object
        ) -> MorrisServiceResult:
            return crossed

    registry = MorrisJobRegistry(CrossedService())
    second_document = {**request_document(), "request_id": "different-request"}
    job_id = registry.create(parse_morris_request(second_document)).job_id
    for _index in range(100):
        envelope = registry.status(job_id)
        if envelope.status == "failed":
            break
        threading.Event().wait(0.001)
    assert envelope.status == "failed"
    assert envelope.report is None
    with pytest.raises(RuntimeError, match="not available"):
        registry.observations(job_id)
    registry.close()


def test_registry_rejects_report_crossed_with_same_request_observations() -> None:
    request = parse_morris_request(request_document())

    def service_with_scale(scale: float) -> RateMorrisService:
        return RateMorrisService(
            evaluator_factory=lambda _design, _config: (
                lambda sample: MorrisEvaluation(
                    "evaluated_hit",
                    {name: scale * float(sample.ordinal) for name in ALL_OUTPUT_NAMES},
                )
            )
        )

    first = service_with_scale(1.0).execute_with_observations(
        request, threading.Event(), lambda _done, _total: None
    )
    second = service_with_scale(2.0).execute_with_observations(
        request, threading.Event(), lambda _done, _total: None
    )
    crossed = MorrisServiceResult(first.report, second.observations)

    class CrossedService:
        def execute(
            self, _request: object, _cancel: object, _progress: object
        ) -> dict[str, object]:
            return crossed.report

        def execute_with_observations(
            self, _request: object, _cancel: object, _progress: object
        ) -> MorrisServiceResult:
            return crossed

    registry = MorrisJobRegistry(CrossedService())
    job_id = registry.create(request).job_id
    for _index in range(100):
        envelope = registry.status(job_id)
        if envelope.status == "failed":
            break
        threading.Event().wait(0.001)
    assert envelope.status == "failed"
    assert envelope.report is None
    with pytest.raises(RuntimeError, match="not available"):
        registry.observations(job_id)
    registry.close()


def test_report_recomputation_does_not_hold_registry_lifecycle_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = parse_morris_request(request_document())
    service = RateMorrisService(
        evaluator_factory=lambda _design, _config: (
            lambda sample: MorrisEvaluation(
                "evaluated_hit",
                {name: float(sample.ordinal) for name in ALL_OUTPUT_NAMES},
            )
        )
    )
    entered = threading.Event()
    release = threading.Event()
    real_analyze = morris_router.analyze_morris

    def delayed_analyze(*args: object, **kwargs: object) -> object:
        entered.set()
        release.wait(timeout=3)
        return real_analyze(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(morris_router, "analyze_morris", delayed_analyze)
    registry = MorrisJobRegistry(service)
    job_id = registry.create(request).job_id
    assert entered.wait(timeout=1)
    assert registry.status(job_id).status == "running"
    release.set()
    for _index in range(100):
        envelope = registry.status(job_id)
        if envelope.status == "completed":
            break
        threading.Event().wait(0.001)
    assert envelope.status == "completed"
    registry.close()


def _await_terminal(client: TestClient, job_id: str) -> dict[str, object]:
    for _index in range(100):
        payload = client.get(f"/authority/morris/jobs/{job_id}").json()
        if payload["status"] in {"completed", "cancelled", "failed"}:
            return payload
    raise AssertionError("job did not finish")
