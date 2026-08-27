"""Headless PyQt coverage for the durable ensemble authority surface."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from rate_of_closure.application.durable_ensemble import (  # noqa: E402
    DURABLE_ENSEMBLE_SCOPE,
    DurableEnsembleCapability,
    DurableEnsembleJobEnvelope,
)
from rate_of_closure.ui.pyqt6.durable_ensemble_tab import (  # noqa: E402
    DurableEnsembleTab,
)
from rate_of_closure.variation import durable_ensemble_evidence  # noqa: E402

from .test_durable_ensemble_authority_contracts import _plan  # noqa: E402
from .test_variation_durable_ensemble_evidence import _summary  # noqa: E402
from .test_variation_simulation_request import _base_config  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _evidence(tmp_path: Path):  # type: ignore[no-untyped-def]
    summary = _summary(tmp_path)
    archive = replace(
        summary.archive,
        status="complete",
        trial_count=3,
        next_index=3,
        elapsed_s=1.0,
    )
    return durable_ensemble_evidence(replace(summary, archive=archive))


class _Client:
    def __init__(self, tmp_path: Path) -> None:
        self.created = []
        self._evidence = _evidence(tmp_path)
        self._request_id = "unset"
        self._archive_id = "unset"

    def capability(self) -> DurableEnsembleCapability:
        assert DURABLE_ENSEMBLE_SCOPE.endswith("/v1")
        return DurableEnsembleCapability(True, "/api/rate-of-closure/v1")

    def create(self, request):  # type: ignore[no-untyped-def]
        self.created.append(request)
        self._request_id = request.request_id
        self._archive_id = request.archive_id
        return self._job("queued", None)

    def status(self, _job_id: str) -> DurableEnsembleJobEnvelope:
        return self._job("completed", self._evidence)

    def cancel(self, _job_id: str) -> DurableEnsembleJobEnvelope:
        return self._job("cancelled", self._evidence)

    def _job(self, status, evidence):  # type: ignore[no-untyped-def]
        completed = 3 if evidence is not None else 0
        return DurableEnsembleJobEnvelope(
            "job-1",
            self._request_id,
            self._archive_id,
            status,
            completed,
            3,
            False,
            evidence,
            None,
        )


def test_tab_runs_shared_plan_and_renders_path_free_moments(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    client = _Client(tmp_path)
    widget = DurableEnsembleTab(client, _plan, poll_interval_ms=1)
    qtbot.addWidget(widget)
    widget.set_simulation_config(_base_config())

    qtbot.waitUntil(lambda: widget._run_button.isEnabled(), timeout=5_000)
    widget._archive_id.setText("review-campaign")
    widget._run_button.click()
    qtbot.waitUntil(lambda: widget._results.rowCount() > 0, timeout=5_000)

    assert len(client.created) == 1
    assert client.created[0].archive_id == "review-campaign"
    assert widget._progress.value() == 3
    assert "completed" in widget._status.text()
    assert widget._results.item(0, 1).text() == "2"


def test_tab_is_explicitly_unavailable_without_local_authority(qtbot) -> None:  # type: ignore[no-untyped-def]
    widget = DurableEnsembleTab(None, _plan)
    qtbot.addWidget(widget)

    assert not widget._run_button.isEnabled()
    assert "unavailable" in widget._status.text().lower()


def test_chunk_size_editor_matches_shared_authority_contract(qtbot) -> None:  # type: ignore[no-untyped-def]
    widget = DurableEnsembleTab(None, _plan)
    qtbot.addWidget(widget)

    assert widget._chunk_size.minimum() == 1
    assert widget._chunk_size.maximum() == 4096
