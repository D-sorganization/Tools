"""Regression tests for InspectorSidebar._apply_changes (#3320).

An Operator clicking 'Apply' after touching a (read-only) safety/PID spin box
previously hit ``QDoubleSpinBox.isModified()`` — a QLineEdit-only method — which
raised ``AttributeError`` inside a Qt slot and, since PyQt 5.5, aborts the whole
HMI via qFatal. The fix compares the spin box value against the last-loaded
baseline instead. These tests assert the slot runs to completion and surfaces a
normal 'Access Denied' dialog rather than crashing.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("requests")  # InspectorSidebar -> workers -> requests

from PyQt6.QtWidgets import QDoubleSpinBox  # noqa: E402

from p1am_control_system.desktop import sidebar as sidebar_module  # noqa: E402
from p1am_control_system.desktop.sidebar import InspectorSidebar  # noqa: E402


def test_qdoublespinbox_has_no_is_modified() -> None:
    """Guard: the bug was calling a method that does not exist on the widget."""
    assert not hasattr(QDoubleSpinBox, "isModified")


@pytest.mark.gui
def test_operator_apply_after_edit_shows_dialog_without_crash(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str]] = []

    def _fake_critical(_parent, title, text, *args, **kwargs):  # noqa: ANN001
        calls.append((title, text))
        return None

    monkeypatch.setattr(sidebar_module.QMessageBox, "critical", _fake_critical)

    widget = InspectorSidebar()
    widget.set_role("Operator")
    widget.selected_tag_id = 0
    # Establish a baseline, then simulate the operator nudging the read-only
    # low-limit spin box above it.
    widget._baseline_low_limit = 10.0
    widget.spin_low_limit.setValue(25.0)

    # Must not raise AttributeError (the pre-fix crash) ...
    widget._apply_changes()

    # ... and must surface the access-denied dialog exactly once.
    assert len(calls) == 1
    assert "Access Denied" in calls[0][0]


@pytest.mark.gui
def test_operator_apply_without_edit_shows_no_dialog(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        sidebar_module.QMessageBox,
        "critical",
        lambda *a, **k: calls.append(("critical", "")),
    )

    widget = InspectorSidebar()
    widget.set_role("Operator")
    widget.selected_tag_id = 0
    widget._baseline_low_limit = 10.0
    widget._baseline_pid_setpoint = 5.0
    widget.spin_low_limit.setValue(10.0)
    widget.spin_pid_sp.setValue(5.0)

    widget._apply_changes()

    assert calls == []
