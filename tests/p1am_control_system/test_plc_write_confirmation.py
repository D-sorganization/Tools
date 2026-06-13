"""Tests for PLC-write confirmation + Control-tab role gating (#3323).

Plant-affecting HMI writes (PID gain apply, tuning start/step, NVRAM routing
deploy, raw tag force override) previously executed on a single click with no
confirmation, and the Control tab had no role gate at all. These tests assert:

- Each write is guarded by a ``QMessageBox.question`` dialog; answering *No*
  fires no HTTP worker.
- Control-tab tuning/gain writes are Admin-only; an Operator is rejected before
  any dialog or HTTP call.
- Answering *Yes* (as Admin) does start the worker.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("requests")
pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QMessageBox  # noqa: E402

from p1am_control_system.desktop import control_tab as control_tab_module  # noqa: E402
from p1am_control_system.desktop import routing_tab as routing_tab_module  # noqa: E402
from p1am_control_system.desktop import sidebar as sidebar_module  # noqa: E402
from p1am_control_system.desktop.control_tab import ControlTab  # noqa: E402
from p1am_control_system.desktop.routing_tab import RoutingTab  # noqa: E402
from p1am_control_system.desktop.sidebar import InspectorSidebar  # noqa: E402


class _SpyWorker:
    """Drop-in replacement for HttpWorker that records construction + start().

    Never touches the network; ``success``/``error`` expose ``.connect`` no-ops
    so call sites that wire callbacks keep working.
    """

    instances: list[_SpyWorker] = []

    class _Signal:
        def connect(self, *_a, **_k) -> None:
            return None

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.started = False
        self.success = self._Signal()
        self.error = self._Signal()
        _SpyWorker.instances.append(self)

    def start(self) -> None:
        self.started = True


@pytest.fixture
def spy_worker(monkeypatch: pytest.MonkeyPatch):
    """Patch HttpWorker in all three desktop modules with a recording spy."""
    _SpyWorker.instances = []
    for mod in (control_tab_module, routing_tab_module, sidebar_module):
        monkeypatch.setattr(mod, "HttpWorker", _SpyWorker)
    return _SpyWorker


@pytest.fixture(autouse=True)
def _robust_pyqtgraph(monkeypatch: pytest.MonkeyPatch):
    """Make ControlTab construction independent of the installed pyqtgraph.

    ControlTab builds pyqtgraph plots in ``__init__`` using ``pg.mkPen`` with
    ``Qt.GlobalColor`` enum colors, whose acceptance varies across pyqtgraph
    releases. These role/confirmation tests only exercise the HTTP-write slots,
    not the plotting, so stub ``mkPen`` to keep construction stable everywhere.
    """
    monkeypatch.setattr(
        control_tab_module.pg, "mkPen", lambda *a, **k: None, raising=False
    )


def _patch_question(monkeypatch: pytest.MonkeyPatch, module, answer) -> list[str]:
    """Patch ``module.QMessageBox.question`` to return *answer*; record titles."""
    titles: list[str] = []

    def _question(_parent, title, _text, *_a, **_k):  # noqa: ANN001
        titles.append(title)
        return answer

    monkeypatch.setattr(module.QMessageBox, "question", _question)
    return titles


def _silence_dialogs(monkeypatch: pytest.MonkeyPatch, module) -> None:
    for name in ("critical", "information", "warning"):
        monkeypatch.setattr(module.QMessageBox, name, lambda *a, **k: None)


# ---------------------------------------------------------------------------
# Control tab: role gating
# ---------------------------------------------------------------------------


@pytest.mark.gui
def test_control_tab_defaults_to_operator(qapp) -> None:
    tab = ControlTab()
    assert tab.user_role == "Operator"


@pytest.mark.gui
def test_set_role_admin_enables_start_tuning(qapp) -> None:
    tab = ControlTab()
    tab.set_role("Admin")
    assert tab.user_role == "Admin"
    assert tab.btn_start_tuning.isEnabled() is True


@pytest.mark.gui
def test_set_role_operator_disables_start_tuning(qapp) -> None:
    tab = ControlTab()
    tab.set_role("Admin")
    tab.set_role("Operator")
    assert tab.btn_start_tuning.isEnabled() is False


@pytest.mark.gui
def test_operator_apply_gains_rejected(
    qapp, monkeypatch: pytest.MonkeyPatch, spy_worker
) -> None:
    tab = ControlTab()
    tab.set_role("Operator")

    class _Cfg:
        pids = [type("P", (), {"kp": 0.0, "ki": 0.0, "kd": 0.0})()]

        def dict(self):
            return {}

    tab.routing_config = _Cfg()
    tab.lbl_recom_kp.setText("1.0")
    tab.lbl_recom_ki.setText("2.0")
    tab.lbl_recom_kd.setText("3.0")

    _silence_dialogs(monkeypatch, control_tab_module)
    # Even if a dialog were shown, answering Yes must not matter: role wins.
    _patch_question(monkeypatch, control_tab_module, QMessageBox.StandardButton.Yes)

    tab._apply_recommended_gains()
    assert spy_worker.instances == []


@pytest.mark.gui
def test_admin_apply_gains_declined_fires_no_worker(
    qapp, monkeypatch: pytest.MonkeyPatch, spy_worker
) -> None:
    tab = ControlTab()
    tab.set_role("Admin")

    class _Cfg:
        pids = [type("P", (), {"kp": 0.0, "ki": 0.0, "kd": 0.0})()]

        def dict(self):
            return {}

    tab.routing_config = _Cfg()
    tab.lbl_recom_kp.setText("1.0")
    tab.lbl_recom_ki.setText("2.0")
    tab.lbl_recom_kd.setText("3.0")

    titles = _patch_question(
        monkeypatch, control_tab_module, QMessageBox.StandardButton.No
    )

    tab._apply_recommended_gains()
    assert titles == ["Confirm PLC write"]
    assert spy_worker.instances == []


@pytest.mark.gui
def test_admin_apply_gains_confirmed_starts_worker(
    qapp, monkeypatch: pytest.MonkeyPatch, spy_worker
) -> None:
    tab = ControlTab()
    tab.set_role("Admin")

    class _Cfg:
        pids = [type("P", (), {"kp": 0.0, "ki": 0.0, "kd": 0.0})()]

        def dict(self):
            return {}

    tab.routing_config = _Cfg()
    tab.lbl_recom_kp.setText("1.0")
    tab.lbl_recom_ki.setText("2.0")
    tab.lbl_recom_kd.setText("3.0")

    _silence_dialogs(monkeypatch, control_tab_module)
    _patch_question(monkeypatch, control_tab_module, QMessageBox.StandardButton.Yes)

    tab._apply_recommended_gains()
    assert len(spy_worker.instances) == 1
    assert spy_worker.instances[0].started is True


@pytest.mark.gui
def test_operator_start_tuning_rejected(
    qapp, monkeypatch: pytest.MonkeyPatch, spy_worker
) -> None:
    tab = ControlTab()
    tab.set_role("Operator")
    _silence_dialogs(monkeypatch, control_tab_module)
    _patch_question(monkeypatch, control_tab_module, QMessageBox.StandardButton.Yes)
    tab._start_tuning()
    assert spy_worker.instances == []


@pytest.mark.gui
def test_admin_apply_step_declined_fires_no_worker(
    qapp, monkeypatch: pytest.MonkeyPatch, spy_worker
) -> None:
    tab = ControlTab()
    tab.set_role("Admin")
    titles = _patch_question(
        monkeypatch, control_tab_module, QMessageBox.StandardButton.No
    )
    tab._apply_step()
    assert titles == ["Confirm PLC write"]
    assert spy_worker.instances == []


# ---------------------------------------------------------------------------
# Routing tab: NVRAM deploy confirmation
# ---------------------------------------------------------------------------


@pytest.mark.gui
def test_routing_deploy_declined_fires_no_worker(
    qapp, monkeypatch: pytest.MonkeyPatch, spy_worker
) -> None:
    tab = RoutingTab()
    tab.set_role("Admin")
    tab.routing_config = object()  # truthy; deploy is blocked before use
    titles = _patch_question(
        monkeypatch, routing_tab_module, QMessageBox.StandardButton.No
    )
    tab._deploy_config()
    assert titles == ["Confirm PLC write"]
    assert spy_worker.instances == []


# ---------------------------------------------------------------------------
# Sidebar: tag force override confirmation (Operator allowed by design)
# ---------------------------------------------------------------------------


@pytest.mark.gui
def test_force_override_declined_fires_no_worker(
    qapp, monkeypatch: pytest.MonkeyPatch, spy_worker
) -> None:
    widget = InspectorSidebar()
    widget.set_role("Operator")
    widget.selected_tag_id = 3
    widget.chk_force_active.setChecked(True)
    titles = _patch_question(monkeypatch, sidebar_module, QMessageBox.StandardButton.No)
    widget._apply_changes()
    assert titles == ["Confirm PLC write"]
    # No force worker started.
    assert all("tags" not in str(w.args) for w in spy_worker.instances)


@pytest.mark.gui
def test_force_override_confirmed_starts_worker(
    qapp, monkeypatch: pytest.MonkeyPatch, spy_worker
) -> None:
    widget = InspectorSidebar()
    widget.set_role("Operator")
    widget.selected_tag_id = 3
    widget.chk_force_active.setChecked(True)
    _patch_question(monkeypatch, sidebar_module, QMessageBox.StandardButton.Yes)
    widget._apply_changes()
    assert any("tags" in str(w.args) for w in spy_worker.instances)
