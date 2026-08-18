"""Headless PyQt coverage for Morris workspace persistence."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from rate_of_closure.application.morris.workspace import (  # noqa: E402
    dumps_morris_workspace,
    parse_morris_workspace,
)
from rate_of_closure.club import get_club  # noqa: E402
from rate_of_closure.model import ImpactScenario  # noqa: E402
from rate_of_closure.simulation import (  # noqa: E402
    BallSetup,
    BallSupportMode,
    ContactMode,
    SimulationConfig,
)
from rate_of_closure.ui.pyqt6.morris_tab import MorrisScreeningTab  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

FIXTURE = Path(__file__).parent / "fixtures" / "morris_workspace_v1.json"


class _BlockingClient:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.cancelled: list[str] = []
        self.created: list[object] = []

    def capability(self):  # type: ignore[no-untyped-def]
        from rate_of_closure.application.morris._response_types import MorrisCapability

        return MorrisCapability(
            True,
            "/api/rate-of-closure/v1",
            "rate-of-closure/morris-request",
            "rate-of-closure/morris-job",
        )

    def create(self, request):  # type: ignore[no-untyped-def]
        from rate_of_closure.application.morris._response_types import MorrisResponseJob

        self.created.append(request)
        return MorrisResponseJob(
            "job-active",
            request.request_id,
            "queued",
            0,
            request.total_samples,
            False,
            None,
            None,
            None,
        )

    def status(self, _job_id: str):  # type: ignore[no-untyped-def]
        from rate_of_closure.application.morris._response_types import MorrisResponseJob

        self.entered.set()
        self.release.wait()
        return MorrisResponseJob(
            "job-active", "active-request", "running", 0, 1, False, None, None, None
        )

    def cancel(self, job_id: str):  # type: ignore[no-untyped-def]
        self.cancelled.append(job_id)


def _fixture_text() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def test_import_restores_all_drafts_controls_and_completed_evidence(qtbot) -> None:  # type: ignore[no-untyped-def]
    workspace = parse_morris_workspace(json.loads(_fixture_text()))
    widget = MorrisScreeningTab(None)
    qtbot.addWidget(widget)
    widget.set_simulation_config(workspace.base_config())

    widget.load_workspace_text(_fixture_text())

    assert widget._trajectories.value() == 12
    assert widget._levels.value() == 4
    assert widget._seed.value() == 73
    assert widget._minimum_effects.value() == 4
    assert widget._workers.value() == 1
    assert tuple(row.workspace_draft() for row in widget._factor_rows) == (
        workspace.setup.factor_drafts
    )
    assert widget._last_job == workspace.completed_evidence.job
    assert widget._results.rowCount() == 10


def test_invalid_or_host_mismatched_import_never_mutates_or_cancels_active_work(
    qtbot,
) -> None:  # type: ignore[no-untyped-def]
    client = _BlockingClient()
    widget = MorrisScreeningTab(client, poll_interval_ms=1)
    qtbot.addWidget(widget)
    qtbot.waitUntil(lambda: widget._run_button.isEnabled(), timeout=2_000)
    widget._run_button.click()
    assert client.entered.wait(timeout=2.0)
    generation = widget._generation
    before = widget.workspace_document()

    invalid = json.loads(_fixture_text())
    invalid["extra"] = True
    with pytest.raises(ValueError, match="fields"):
        widget.load_workspace_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="host base"):
        widget.load_workspace_text(_fixture_text())

    assert widget._generation == generation
    assert widget.workspace_document() == before
    assert not client.cancelled
    client.release.set()
    qtbot.waitUntil(lambda: not widget.has_running_workers(), timeout=2_000)


def test_valid_import_invalidates_active_work_only_after_complete_parse(qtbot) -> None:  # type: ignore[no-untyped-def]
    client = _BlockingClient()
    workspace = parse_morris_workspace(json.loads(_fixture_text()))
    widget = MorrisScreeningTab(client, poll_interval_ms=1)
    qtbot.addWidget(widget)
    widget.set_simulation_config(workspace.base_config())
    qtbot.waitUntil(lambda: widget._run_button.isEnabled(), timeout=2_000)
    widget._run_button.click()
    assert client.entered.wait(timeout=2.0)
    generation = widget._generation

    widget.load_workspace_text(_fixture_text())

    assert widget._generation == generation + 1
    assert widget._last_job == workspace.completed_evidence.job
    assert widget._results.rowCount() == 10
    client.release.set()
    qtbot.waitUntil(lambda: not widget.has_running_workers(), timeout=2_000)


def test_ground_workspace_retains_disabled_canonical_tee_draft(qtbot) -> None:  # type: ignore[no-untyped-def]
    widget = MorrisScreeningTab(None)
    qtbot.addWidget(widget)
    widget.set_simulation_config(
        SimulationConfig(
            scenario=ImpactScenario(113.0),
            club=get_club("7-Iron"),
            ball_setup=BallSetup(BallSupportMode.GROUND, 0.0),
            source_kind="double_pendulum",
            contact_mode=ContactMode.FIXED_BALL_CONTACT,
        )
    )

    workspace = widget.workspace_document()
    parsed = parse_morris_workspace(json.loads(dumps_morris_workspace(workspace)))

    assert len(widget._factor_rows) == 9
    assert len(parsed.setup.factor_drafts) == 10
    tee = parsed.setup.factor_drafts[-1]
    assert not tee.enabled
    assert tee.validation_error is None


def test_disabled_invalid_raw_draft_is_lossless_through_pyqt(qtbot) -> None:  # type: ignore[no-untyped-def]
    document = json.loads(_fixture_text())
    document["completed_evidence"] = None
    draft = document["setup"]["factor_drafts"][0]
    draft.update(
        enabled=False,
        lower="not-a-number",
        validation_error="Bounds must be finite numbers with lower < upper.",
    )
    text = json.dumps(document)
    workspace = parse_morris_workspace(document)
    widget = MorrisScreeningTab(None)
    qtbot.addWidget(widget)
    widget.set_simulation_config(workspace.base_config())

    widget.load_workspace_text(text)

    assert json.loads(dumps_morris_workspace(widget.workspace_document())) == document


def test_imported_invalid_draft_cannot_submit_stale_numeric_value(qtbot) -> None:  # type: ignore[no-untyped-def]
    document = json.loads(_fixture_text())
    document["completed_evidence"] = None
    draft = document["setup"]["factor_drafts"][0]
    draft.update(
        enabled=False,
        lower="not-a-number",
        validation_error="Bounds must be finite numbers with lower < upper.",
    )
    workspace = parse_morris_workspace(document)
    client = _BlockingClient()
    widget = MorrisScreeningTab(client, poll_interval_ms=1)
    qtbot.addWidget(widget)
    widget.set_simulation_config(workspace.base_config())
    qtbot.waitUntil(lambda: widget._run_button.isEnabled(), timeout=2_000)
    widget.load_workspace_text(json.dumps(document))
    row = widget._factor_rows[0]

    row.enabled.setChecked(True)
    widget._start_run()

    assert not client.created
    assert "cannot be enabled until its bounds are valid" in widget._status.text()

    row.lower_editor.setValue(-2.0)
    widget._start_run()
    qtbot.waitUntil(lambda: bool(client.created), timeout=2_000)
    assert client.created[0].factors[0].lower == -2.0
    client.release.set()
    qtbot.waitUntil(lambda: not widget.has_running_workers(), timeout=2_000)


@pytest.mark.parametrize(
    ("field", "value"),
    (("seed", 2**31), ("trajectories", 5_001)),
)
def test_unrepresentable_controls_fail_before_active_run_mutation(
    qtbot, field: str, value: int
) -> None:  # type: ignore[no-untyped-def]
    client = _BlockingClient()
    workspace = parse_morris_workspace(json.loads(_fixture_text()))
    widget = MorrisScreeningTab(client, poll_interval_ms=1)
    qtbot.addWidget(widget)
    widget.set_simulation_config(workspace.base_config())
    qtbot.waitUntil(lambda: widget._run_button.isEnabled(), timeout=2_000)
    widget._run_button.click()
    assert client.entered.wait(timeout=2.0)
    generation = widget._generation
    document = json.loads(_fixture_text())
    document["setup"][field] = value
    document["completed_evidence"] = None

    with pytest.raises(ValueError):
        widget.load_workspace_text(json.dumps(document))

    assert widget._generation == generation
    assert not client.cancelled
    client.release.set()
    qtbot.waitUntil(lambda: not widget.has_running_workers(), timeout=2_000)
