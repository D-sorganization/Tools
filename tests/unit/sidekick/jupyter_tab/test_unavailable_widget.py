"""Tests for the unavailable-state placeholder widget."""

from __future__ import annotations

import pytest

try:
    from upstream_drift_tools.ui.tools_sidebar.qt_compat import (
        QT_API,
        QtWidgets,
    )
except ImportError:  # pragma: no cover
    QT_API = ""
    QtWidgets = None  # type: ignore[assignment]

from upstream_drift_tools.ui.tools_sidebar.jupyter_tab.unavailable_widget import (  # noqa: E402
    JupyterUnavailableWidget,
)


@pytest.fixture
def qt_app() -> object:
    if QtWidgets is None or QT_API == "":
        pytest.skip("Qt widgets unavailable")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return app


def test_unavailable_widget_contains_install_hint(qt_app: object) -> None:
    _ = qt_app
    widget = JupyterUnavailableWidget(install_hint="pip install .[jupyter]")
    label = widget.findChild(QtWidgets.QLabel, "SidekickJupyterInstallHint")
    assert label is not None
    assert "pip install" in label.text()


def test_unavailable_widget_has_copy_button(qt_app: object) -> None:
    _ = qt_app
    widget = JupyterUnavailableWidget(install_hint="pip install .[jupyter]")
    button = widget.findChild(QtWidgets.QPushButton, "SidekickJupyterInstallCopy")
    assert button is not None
