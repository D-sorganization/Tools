"""Regression test for ControlTab MPC plot construction (#3323-followup).

The MPC comparison plots in ``ControlTab._init_ui`` passed ``Qt.GlobalColor``
enum members straight into ``pg.mkPen(color=...)``. Under pyqtgraph 0.13.7+ /
0.14.0 with PyQt6, ``mkColor`` raises
``TypeError: Not sure how to make a color from "(<GlobalColor.red: 7>,)"``,
which aborts ``ControlTab()`` construction entirely — so any test or launch that
builds the Control tab dies. The fix passes pyqtgraph-accepted color forms
(``"r"`` and the ``(0, 100, 0)`` darkGreen tuple) instead. This test simply
constructs the widget and asserts the four MPC curve attributes exist, which
would have caught the crash.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")
pytest.importorskip("requests")  # control_tab -> workers -> requests

from p1am_control_system.desktop.control_tab import ControlTab  # noqa: E402


@pytest.mark.gui
def test_control_tab_constructs_with_mpc_curves(qapp) -> None:
    """ControlTab() builds without a mkColor TypeError and wires up its curves."""
    widget = ControlTab()

    for attr in ("curve_pid_pv", "curve_mpc_pv", "curve_pid_cv", "curve_mpc_cv"):
        assert hasattr(widget, attr), f"missing MPC curve attribute: {attr}"
