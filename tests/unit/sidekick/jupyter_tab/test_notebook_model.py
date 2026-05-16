"""Tests for the immutable ``NotebookDocument`` data model."""

from __future__ import annotations

import pytest
from upstream_drift_tools.ui.tools_sidebar.jupyter_tab.notebook_model import (
    CellOutput,
    CodeCell,
    MarkdownCell,
    NotebookDocument,
    RawCell,
)


def test_notebook_document_is_immutable() -> None:
    doc = NotebookDocument(cells=(MarkdownCell(source="hi"),))
    with pytest.raises((AttributeError, TypeError)):
        doc.cells = ()  # type: ignore[misc]


def test_cells_tuple_cannot_be_replaced_in_place() -> None:
    cell = MarkdownCell(source="hi")
    doc = NotebookDocument(cells=(cell,))
    assert isinstance(doc.cells, tuple)


def test_cell_types_are_distinct() -> None:
    md = MarkdownCell(source="text")
    code = CodeCell(source="x = 1", outputs=(CellOutput(text="1"),))
    raw = RawCell(source="raw text")
    assert md.cell_type == "markdown"
    assert code.cell_type == "code"
    assert raw.cell_type == "raw"
    assert {type(md), type(code), type(raw)} == {MarkdownCell, CodeCell, RawCell}
