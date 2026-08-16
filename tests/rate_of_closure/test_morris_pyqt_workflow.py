"""Headless PyQt coverage for the authority-backed Morris workflow."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import replace

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from PyQt6.QtWidgets import QTableWidget  # noqa: E402

from rate_of_closure.application.morris._response_types import (  # noqa: E402
    MorrisCapability,
    MorrisDenominator,
    MorrisEffects,
    MorrisResponseEstimate,
    MorrisResponseJob,
    MorrisResponseReport,
    MorrisSource,
    MorrisTarget,
)
from rate_of_closure.application.morris.request_document import (  # noqa: E402
    CANONICAL_MORRIS_FACTOR_KEYS,
)
from rate_of_closure.club import get_club  # noqa: E402
from rate_of_closure.model import ImpactScenario  # noqa: E402
from rate_of_closure.simulation import SimulationConfig  # noqa: E402
from rate_of_closure.simulation.contact import ContactMode  # noqa: E402
from rate_of_closure.ui.pyqt6.main_window import (  # noqa: E402
    RateOfClosureMainWindow,
)
from rate_of_closure.ui.pyqt6.morris_tab import MorrisScreeningTab  # noqa: E402
from rate_of_closure.ui.pyqt6.variation_tab import VariationTab  # noqa: E402
from rate_of_closure.ui.pyqt6.variation_workspace import (  # noqa: E402
    VariationWorkspace,
)
from shared.python.swing_sim.types import PlaneOrientation  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _job(
    status: str,
    *,
    report: MorrisResponseReport | None = None,
    job_id: str = "job-1",
    request_id: str = "request-1",
) -> MorrisResponseJob:
    total = 20
    return MorrisResponseJob(
        job_id,
        request_id,
        status,  # type: ignore[arg-type]
        total if status == "completed" else 2,
        total,
        False,
        report,
        None,
        None,
    )


def _report() -> MorrisResponseReport:
    source = MorrisSource(
        CANONICAL_MORRIS_FACTOR_KEYS[0],
        CANONICAL_MORRIS_FACTOR_KEYS[0],
        "deg",
        (-2.0, 2.0),
        None,
        (),
    )
    target = MorrisTarget("carry_m", "m", "scalar", None, None, None)
    estimate = MorrisResponseEstimate(
        source,
        target,
        MorrisEffects(1.5, 2.0, 0.25, 0.5),
        "available",
        "adequate",
        MorrisDenominator(12, 12, 1, 0, 0, 0),
    )
    return MorrisResponseReport(
        12,
        4,
        7,
        24,
        2.0 / 3.0,
        ("test assumption",),
        "Interactions are screened, not decomposed.",
        (estimate,),
    )


def _config() -> SimulationConfig:
    return SimulationConfig(
        scenario=ImpactScenario(clubhead_speed_mph=113.0),
        club=get_club("Driver 10.5°"),
        source_kind="double_pendulum",
        contact_mode=ContactMode.FIXED_BALL_CONTACT,
    )


class _FakeClient:
    def __init__(self) -> None:
        self.created = []
        self.cancelled: list[str] = []
        self._request_id = "request-not-created"
        self._states = deque(("running", "completed"))

    def capability(self) -> MorrisCapability:
        return MorrisCapability(
            True,
            "/api/rate-of-closure/v1",
            "rate-of-closure/morris-request",
            "rate-of-closure/morris-job",
        )

    def create(self, request):  # type: ignore[no-untyped-def]
        self.created.append(request)
        self._request_id = request.request_id
        return _job("queued", request_id=self._request_id)

    def status(self, _job_id: str) -> MorrisResponseJob:
        status = self._states.popleft()
        report = _report() if status == "completed" else None
        return _job(status, report=report, request_id=self._request_id)

    def cancel(self, job_id: str) -> MorrisResponseJob:
        self.cancelled.append(job_id)
        return _job("cancelled", request_id=self._request_id)


class _CancelClient(_FakeClient):
    def status(self, _job_id: str) -> MorrisResponseJob:
        return _job("running", request_id=self._request_id)


class _IdentityMismatchClient(_FakeClient):
    def __init__(self, phase: str) -> None:
        super().__init__()
        self.phase = phase

    def create(self, request):  # type: ignore[no-untyped-def]
        job = super().create(request)
        if self.phase == "create-request":
            return replace(job, request_id="crossed-request")
        return job

    def status(self, job_id: str) -> MorrisResponseJob:
        if self.phase.startswith("cancel"):
            return _job("running", request_id=self._request_id)
        job = super().status(job_id)
        if self.phase == "status-request":
            return replace(job, request_id="crossed-request")
        if self.phase == "status-job":
            return replace(job, job_id="crossed-job")
        return job

    def cancel(self, job_id: str) -> MorrisResponseJob:
        job = super().cancel(job_id)
        if self.phase == "cancel-request":
            return replace(job, request_id="crossed-request")
        if self.phase == "cancel-job":
            return replace(job, job_id="crossed-job")
        return job


class _BlockingClient(_FakeClient):
    def __init__(self, blocked_call: str) -> None:
        super().__init__()
        self.blocked_call = blocked_call
        self.entered = threading.Event()
        self.release = threading.Event()

    def _block(self, call: str) -> None:
        if self.blocked_call == call:
            self.entered.set()
            self.release.wait()

    def capability(self) -> MorrisCapability:
        self._block("capability")
        return super().capability()

    def create(self, request):  # type: ignore[no-untyped-def]
        self._block("create")
        return super().create(request)

    def status(self, job_id: str) -> MorrisResponseJob:
        self._block("status")
        return super().status(job_id)


def test_workspace_keeps_monte_carlo_as_a_distinct_sibling(qtbot) -> None:  # type: ignore[no-untyped-def]
    monte_carlo = VariationTab()
    morris = MorrisScreeningTab(None)
    workspace = VariationWorkspace(monte_carlo, morris)
    qtbot.addWidget(workspace)

    assert workspace.tabs().tabText(0) == "Monte Carlo & Dispersion"
    assert workspace.tabs().tabText(1) == "Morris Screening"
    assert workspace.tabs().widget(0) is monte_carlo
    assert not morris._run_button.isEnabled()
    assert "unavailable" in morris._status.text().lower()


def test_capability_enables_ordered_editable_factors_and_run(qtbot) -> None:  # type: ignore[no-untyped-def]
    client = _FakeClient()
    widget = MorrisScreeningTab(client, poll_interval_ms=1)
    qtbot.addWidget(widget)

    qtbot.waitUntil(lambda: widget._run_button.isEnabled(), timeout=2_000)
    assert tuple(row.variable_key for row in widget._factor_rows) == (
        CANONICAL_MORRIS_FACTOR_KEYS
    )
    assert all(
        row.lower_editor.isEnabled() and row.upper_editor.isEnabled()
        for row in widget._factor_rows
    )

    widget._trajectories.setValue(12)
    widget._levels.setValue(6)
    widget._seed.setValue(7)
    widget._minimum_effects.setValue(4)
    widget._workers.setValue(1)
    widget._run_button.click()

    qtbot.waitUntil(lambda: widget._results.rowCount() == 1, timeout=5_000)
    assert len(client.created) == 1
    request = client.created[0]
    assert request.levels == 6 and request.seed == 7
    assert widget._target_combo.currentData() == "carry_m"
    assert widget._results.item(0, 2).text() == "2"
    assert widget._results.item(0, 3).text() == "0.25"
    assert "12/12 valid" in widget._results.item(0, 7).text()


def test_stale_worker_updates_are_ignored(qtbot) -> None:  # type: ignore[no-untyped-def]
    widget = MorrisScreeningTab(None)
    qtbot.addWidget(widget)
    original = widget._status.text()

    widget._accept_job(widget._generation + 1, _job("running"))

    assert widget._status.text() == original


def test_cancel_is_sent_once_and_reaches_terminal_state(qtbot) -> None:  # type: ignore[no-untyped-def]
    client = _CancelClient()
    widget = MorrisScreeningTab(client, poll_interval_ms=1)
    qtbot.addWidget(widget)
    qtbot.waitUntil(lambda: widget._run_button.isEnabled(), timeout=2_000)
    widget._run_button.click()
    qtbot.waitUntil(lambda: bool(client.created), timeout=2_000)

    widget._cancel_button.click()

    qtbot.waitUntil(
        lambda: widget._status.text() == "Morris study cancelled", timeout=2_000
    )
    assert client.cancelled == ["job-1"]


def test_incompatible_base_is_disabled_without_dropping_semantics(qtbot) -> None:  # type: ignore[no-untyped-def]
    client = _FakeClient()
    widget = MorrisScreeningTab(client)
    qtbot.addWidget(widget)
    widget.set_simulation_config(
        SimulationConfig(
            scenario=ImpactScenario(clubhead_speed_mph=113.0),
            club=get_club("Driver 10.5°"),
            source_kind="manual",
        )
    )
    qtbot.waitUntil(lambda: widget._capability_worker is None, timeout=2_000)

    assert not widget._run_button.isEnabled()
    assert "custom torque/run semantics" in widget._status.text()


@pytest.mark.parametrize("blocked_call", ("capability", "create", "status"))
def test_shutdown_never_drops_a_running_transport_thread(
    qtbot, blocked_call: str
) -> None:  # type: ignore[no-untyped-def]
    client = _BlockingClient(blocked_call)
    widget = MorrisScreeningTab(client, poll_interval_ms=1)
    qtbot.addWidget(widget)
    if blocked_call != "capability":
        qtbot.waitUntil(lambda: widget._run_button.isEnabled(), timeout=2_000)
        widget._run_button.click()
    assert client.entered.wait(timeout=2.0)

    started = time.monotonic()
    complete = widget.stop()

    assert time.monotonic() - started < 0.2
    assert not complete
    assert widget.has_running_workers()
    assert (
        widget._capability_worker is not None
        or widget._worker is not None
        or widget._retired_workers
    )

    client.release.set()
    qtbot.waitUntil(lambda: not widget.has_running_workers(), timeout=2_000)


def test_active_run_config_change_invalidates_old_result_and_controls(qtbot) -> None:  # type: ignore[no-untyped-def]
    client = _BlockingClient("status")
    widget = MorrisScreeningTab(client, poll_interval_ms=1)
    qtbot.addWidget(widget)
    qtbot.waitUntil(lambda: widget._run_button.isEnabled(), timeout=2_000)
    widget._run_button.click()
    assert client.entered.wait(timeout=2.0)
    prior_generation = widget._generation
    replacement = SimulationConfig(
        scenario=ImpactScenario(clubhead_speed_mph=113.0),
        club=get_club("Driver 10.5°"),
        source_kind="double_pendulum",
        contact_mode=ContactMode.FIXED_BALL_CONTACT,
        plane=PlaneOrientation(yaw_deg=5.0),
    )

    widget.set_simulation_config(replacement)

    assert widget._generation == prior_generation + 1
    assert widget.simulation_config() == replacement
    assert not widget._run_button.isEnabled()
    assert not widget._factor_rows[0].isEnabled()
    assert widget._results.rowCount() == 0
    assert widget._target_combo.count() == 0
    assert widget._progress.value() == 0
    assert "cancelling the prior" in widget._status.text().lower()

    client.release.set()
    qtbot.waitUntil(lambda: not widget.has_running_workers(), timeout=2_000)
    assert widget._results.rowCount() == 0
    assert widget._run_button.isEnabled()
    assert "authority ready" in widget._status.text().lower()


def test_shutdown_does_not_call_qthread_wait(qtbot, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    client = _BlockingClient("create")
    widget = MorrisScreeningTab(client, poll_interval_ms=1)
    qtbot.addWidget(widget)
    qtbot.waitUntil(lambda: widget._run_button.isEnabled(), timeout=2_000)
    widget._run_button.click()
    assert client.entered.wait(timeout=2.0)
    assert widget._worker is not None

    def fail_wait(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("GUI shutdown must not wait synchronously")

    monkeypatch.setattr(widget._worker, "wait", fail_wait)
    assert not widget.stop()
    client.release.set()
    qtbot.waitUntil(lambda: not widget.has_running_workers(), timeout=2_000)


def test_completed_result_is_invalidated_by_design_and_factor_edits(qtbot) -> None:  # type: ignore[no-untyped-def]
    client = _FakeClient()
    widget = MorrisScreeningTab(client, poll_interval_ms=1)
    qtbot.addWidget(widget)
    qtbot.waitUntil(lambda: widget._run_button.isEnabled(), timeout=2_000)
    widget._run_button.click()
    qtbot.waitUntil(lambda: widget._results.rowCount() == 1, timeout=2_000)

    widget._seed.setValue(widget._seed.value() + 1)

    assert widget._results.rowCount() == 0
    assert widget._target_combo.count() == 0
    assert "inputs changed" in widget._status.text().lower()

    client = _FakeClient()
    other = MorrisScreeningTab(client, poll_interval_ms=1)
    qtbot.addWidget(other)
    qtbot.waitUntil(lambda: other._run_button.isEnabled(), timeout=2_000)
    other._run_button.click()
    qtbot.waitUntil(lambda: other._results.rowCount() == 1, timeout=2_000)

    first = other._factor_rows[0]
    first.lower_editor.setValue(first.lower_editor.value() - 1.0)

    assert other._results.rowCount() == 0
    assert other._target_combo.count() == 0
    assert "inputs changed" in other._status.text().lower()


def test_result_table_is_read_only(qtbot) -> None:  # type: ignore[no-untyped-def]
    widget = MorrisScreeningTab(None)
    qtbot.addWidget(widget)

    assert widget._results.editTriggers() == QTableWidget.EditTrigger.NoEditTriggers


@pytest.mark.parametrize(
    "phase",
    (
        "create-request",
        "status-request",
        "status-job",
        "cancel-request",
        "cancel-job",
    ),
)
def test_worker_fails_closed_when_authority_identity_changes(qtbot, phase: str) -> None:  # type: ignore[no-untyped-def]
    client = _IdentityMismatchClient(phase)
    widget = MorrisScreeningTab(client, poll_interval_ms=1)
    qtbot.addWidget(widget)
    qtbot.waitUntil(lambda: widget._run_button.isEnabled(), timeout=2_000)
    widget._run_button.click()
    qtbot.waitUntil(lambda: bool(client.created), timeout=2_000)
    if phase.startswith("cancel"):
        qtbot.waitUntil(lambda: widget._cancel_button.isEnabled(), timeout=2_000)
        widget._cancel_button.click()

    qtbot.waitUntil(
        lambda: "request failed" in widget._status.text().lower(), timeout=2_000
    )
    assert widget._results.rowCount() == 0


def test_current_simulation_config_rebuilds_presented_factors(qtbot) -> None:  # type: ignore[no-untyped-def]
    widget = MorrisScreeningTab(None)
    qtbot.addWidget(widget)
    config = _config()

    widget.set_simulation_config(config)

    assert widget.simulation_config() == config
    assert widget._factor_rows[0].label.text()
    assert widget._factor_rows[0].lower_editor.toolTip()


def test_main_window_forwards_real_control_edits_to_all_config_consumers(
    qtbot, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        RateOfClosureMainWindow,
        "_initialize_view_content",
        lambda _self: None,
    )
    window = RateOfClosureMainWindow()
    qtbot.addWidget(window)

    window._simulation_tab._flight_combo.setCurrentText("nathan")
    config = window._simulation_tab.config()

    assert window._derivation_view.config().flight_model == "nathan"
    assert window._variation_tab._base_simulation_config == config
    assert window._morris_tab.simulation_config() == config


def test_invalid_prescribed_torque_selection_fails_both_workflows_closed(
    qtbot, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        RateOfClosureMainWindow,
        "_initialize_view_content",
        lambda _self: None,
    )
    window = RateOfClosureMainWindow(morris_client=_FakeClient())
    qtbot.addWidget(window)
    qtbot.waitUntil(lambda: window._morris_tab._run_button.isEnabled(), timeout=2_000)

    window._simulation_tab._torque_profile_panel._run_mode_combo.setCurrentIndex(1)

    assert not window._variation_tab._run_button.isEnabled()
    assert not window._morris_tab._run_button.isEnabled()
    assert "incomplete or invalid" in window._variation_tab._status.text().lower()
    assert "incomplete or invalid" in window._morris_tab._status.text().lower()
