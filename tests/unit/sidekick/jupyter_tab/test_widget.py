# ruff: noqa: E501
"""Tests for ``JupyterNotebookWidget``."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("nbformat")

try:
    from upstream_drift_tools.ui.tools_sidebar.qt_compat import (  # noqa: E402
        QT_API,
        QtWidgets,
    )
except ImportError:  # pragma: no cover
    QT_API = ""
    QtWidgets = None  # type: ignore[assignment]

from upstream_drift_tools.ui.tools_sidebar.jupyter_tab.notebook_loader import (  # noqa: E402
    load_notebook,
)
from upstream_drift_tools.ui.tools_sidebar.jupyter_tab.widget import (  # noqa: E402
    JupyterNotebookWidget,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample.ipynb"


@pytest.fixture
def qt_app() -> object:
    if QtWidgets is None or QT_API == "":
        pytest.skip("Qt widgets unavailable")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return app


def test_widget_constructs_cell_widgets(qt_app: object) -> None:
    _ = qt_app
    document = load_notebook(FIXTURE)
    widget = JupyterNotebookWidget(document=document)
    cell_widgets = widget.cell_widgets()
    assert len(cell_widgets) == 2


def test_widget_code_cell_is_read_only(qt_app: object) -> None:
    _ = qt_app
    document = load_notebook(FIXTURE)
    widget = JupyterNotebookWidget(document=document)
    cell_widgets = widget.cell_widgets()
    code_editor = cell_widgets[1].findChild(QtWidgets.QTextEdit)
    assert code_editor is not None
    assert code_editor.isReadOnly() is True


def test_widget_reads_cells_only_via_document_attribute(qt_app: object) -> None:
    _ = qt_app
    document = load_notebook(FIXTURE)
    widget = JupyterNotebookWidget(document=document)
    assert widget.document is document
