"""Nonblocking, generation-safe PyQt plot computation contracts."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from rate_of_closure.club import get_club  # noqa: E402
from rate_of_closure.model import ImpactScenario  # noqa: E402
from rate_of_closure.plotting import (  # noqa: E402
    PlotData,
    builtin_spec,
    compute_plot_data,
)
from rate_of_closure.simulation import SimulationConfig, run_simulation  # noqa: E402
from rate_of_closure.ui.pyqt6.plots_process_worker import (  # noqa: E402
    PlotComputeProcess,
    PlotDataPayload,
    PlotProcessRequest,
    ProcessPlotOutcome,
)
from rate_of_closure.ui.pyqt6.plots_tab import PlotsTab  # noqa: E402
from rate_of_closure.ui.pyqt6.plots_tab_computation import (  # noqa: E402
    PlotComputationOutcome,
    PlotComputeWorker,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture(scope="module")
def reference_run():  # type: ignore[no-untyped-def]
    return run_simulation(
        SimulationConfig(
            scenario=ImpactScenario(clubhead_speed_mph=113.0),
            club=get_club("Driver 10.5°"),
        )
    )


def _data(spec, marker: float) -> PlotData:  # type: ignore[no-untyped-def]
    return PlotData(
        spec=spec,
        x=np.asarray([0.0, 1.0]),
        series={"Evidence": np.asarray([marker, marker + 1.0])},
        x_label="Input",
        y_label="Evidence",
    )


def test_show_is_nonblocking_and_publishes_complete_worker_result(
    qtbot, monkeypatch, reference_run
) -> None:  # type: ignore[no-untyped-def]
    import rate_of_closure.ui.pyqt6.plots_tab as plots_module

    started = threading.Event()
    release = threading.Event()

    def slow(spec, _run, should_cancel=None):  # type: ignore[no-untyped-def]
        started.set()
        assert release.wait(2.0)
        if should_cancel is not None and should_cancel():
            raise InterruptedError
        return _data(spec, 7.0)

    monkeypatch.setattr(plots_module, "compute_plot_data", slow)
    tab = PlotsTab()
    qtbot.addWidget(tab)
    tab.set_run(reference_run)

    opened = time.perf_counter()
    tab.show()
    # Widget realization is platform-dependent, but must remain well below the
    # blocked worker's 2 s release ceiling.
    assert time.perf_counter() - opened < 0.75
    assert started.wait(1.0)
    assert tab.current_data() is None
    assert "Computing plots" in tab._status.text()
    release.set()
    qtbot.waitUntil(lambda: tab.current_data() is not None, timeout=5_000)
    assert tab.current_data().series["Evidence"][0] == 7.0
    assert tab._status.text() == ""


def test_stale_worker_cannot_publish_after_reference_generation_changes(
    qtbot, monkeypatch, reference_run
) -> None:  # type: ignore[no-untyped-def]
    import rate_of_closure.ui.pyqt6.plots_tab as plots_module

    first_started = threading.Event()
    release_first = threading.Event()
    calls = 0

    def controlled(spec, _run, should_cancel=None):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        marker = float(calls)
        if calls == 1:
            first_started.set()
            assert release_first.wait(2.0)
        if should_cancel is not None and should_cancel():
            raise InterruptedError
        return _data(spec, marker)

    monkeypatch.setattr(plots_module, "compute_plot_data", controlled)
    tab = PlotsTab()
    qtbot.addWidget(tab)
    tab.set_run(reference_run)
    tab.show()
    assert first_started.wait(1.0)

    tab.set_run(reference_run)
    release_first.set()
    qtbot.waitUntil(lambda: calls == 2, timeout=5_000)
    qtbot.waitUntil(lambda: tab.current_data() is not None, timeout=5_000)
    assert tab.current_data().series["Evidence"][0] == 2.0


def test_close_cooperatively_stops_pending_plot_work(
    qtbot, monkeypatch, reference_run
) -> None:  # type: ignore[no-untyped-def]
    import rate_of_closure.ui.pyqt6.plots_tab as plots_module

    started = threading.Event()

    def cancellable(spec, _run, should_cancel=None):  # type: ignore[no-untyped-def]
        started.set()
        while should_cancel is None or not should_cancel():
            time.sleep(0.005)
        raise InterruptedError

    monkeypatch.setattr(plots_module, "compute_plot_data", cancellable)
    tab = PlotsTab()
    qtbot.addWidget(tab)
    tab.set_run(reference_run)
    tab.show()
    assert started.wait(1.0)

    closing = time.perf_counter()
    tab.close()
    assert time.perf_counter() - closing < 0.5
    worker = tab._plot_worker
    assert worker is None or not worker.isRunning()


def test_production_process_publishes_one_complete_immutable_result(
    qtbot, reference_run
) -> None:  # type: ignore[no-untyped-def]
    request = PlotProcessRequest(
        generation=19,
        requests=((0, builtin_spec("swing_time_series", reference_run)),),
        reference_run=reference_run,
        scenario=reference_run.config.scenario,
    )
    worker = PlotComputeProcess(request)
    emitted: list[tuple[int, object, object]] = []
    failures: list[tuple[int, str]] = []
    worker.succeeded.connect(lambda *args: emitted.append(args))
    worker.failed.connect(lambda *args: failures.append(args))

    worker.start()
    qtbot.waitUntil(lambda: worker._finished, timeout=15_000)

    assert failures == []
    assert len(emitted) == 1
    generation, run, outcomes = emitted[0]
    assert generation == 19
    assert type(run) is type(reference_run)
    assert run.config.scenario == reference_run.config.scenario
    assert np.array_equal(run.swing_times, reference_run.swing_times)
    assert len(outcomes) == 1
    assert outcomes[0].row == 0
    assert outcomes[0].error is None
    assert outcomes[0].data is not None
    assert outcomes[0].data.x.flags.writeable is False
    worker.deleteLater()


def test_canonical_sweep_honors_cancellation_before_scientific_work(
    reference_run,
) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(InterruptedError, match="cancelled"):
        compute_plot_data(
            builtin_spec("closure_sweep", reference_run),
            reference_run,
            lambda: True,
        )


def test_process_boundary_rejects_malformed_array_payload(
    reference_run,
) -> None:  # type: ignore[no-untyped-def]
    spec = builtin_spec("swing_time_series", reference_run)
    request = PlotProcessRequest(
        23, ((0, spec),), reference_run, reference_run.config.scenario
    )
    worker = PlotComputeProcess(request)
    failures: list[tuple[int, str]] = []
    worker.failed.connect(lambda *args: failures.append(args))
    malformed = PlotDataPayload(0, spec, b"x", (), "Time", "Evidence")

    worker._publish(
        (
            "success",
            worker.authority_identity,
            23,
            reference_run,
            (ProcessPlotOutcome(0, malformed, None),),
        )
    )

    assert failures
    assert failures[0][0] == 23
    assert "Malformed plot process outcomes" in failures[0][1]
    worker.deleteLater()


def test_process_boundary_rejects_wrong_request_identity(
    reference_run,
) -> None:  # type: ignore[no-untyped-def]
    spec = builtin_spec("swing_time_series", reference_run)
    worker = PlotComputeProcess(
        PlotProcessRequest(
            29, ((0, spec),), reference_run, reference_run.config.scenario
        )
    )
    failures: list[tuple[int, str]] = []
    worker.failed.connect(lambda *args: failures.append(args))

    worker._publish(("success", "0" * 64, 29, reference_run, ()))

    assert failures == [(29, "Malformed plot process result")]
    worker.deleteLater()


@pytest.mark.parametrize("defect", ["subset", "wrong-spec"])
def test_worker_result_must_match_every_requested_pane_before_publication(
    qtbot, reference_run, defect
) -> None:  # type: ignore[no-untyped-def]
    tab = PlotsTab()
    qtbot.addWidget(tab)
    current = tab.current_spec()
    assert current is not None
    other = builtin_spec("swing_time_series", reference_run)
    requests = ((0, current), (1, other)) if defect == "subset" else ((0, current),)
    worker = PlotComputeWorker(
        31,
        requests,
        reference_run,
        reference_run.config.scenario,
        compute_plot_data,
    )
    tab._plot_worker = worker
    tab._plot_generation = 31
    returned = _data(other if defect == "wrong-spec" else current, 5.0)

    tab._accept_plot_success(
        worker,
        31,
        reference_run,
        (PlotComputationOutcome(0, returned, None),),
    )

    assert tab.current_data() is None
    assert "Malformed plot pane" in tab._status.text()
    tab._plot_worker = None
    worker.deleteLater()


def test_unpicklable_process_request_is_a_bounded_first_run_error(
    qtbot, monkeypatch, reference_run
) -> None:  # type: ignore[no-untyped-def]
    import rate_of_closure.ui.pyqt6.plots_process_worker as process_module

    original = process_module.pickle.dumps

    def fail_request(value):  # type: ignore[no-untyped-def]
        if isinstance(value, PlotProcessRequest):
            raise TypeError("unpicklable plot authority")
        return original(value)

    monkeypatch.setattr(process_module.pickle, "dumps", fail_request)
    tab = PlotsTab()
    qtbot.addWidget(tab)
    tab.set_run(reference_run)

    tab.show()

    assert tab._plot_worker is None
    assert "could not start: unpicklable plot authority" in tab._status.text()
    assert tab.current_data() is None
