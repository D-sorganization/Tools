"""Tests for ``notebook_loader``."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("nbformat")

from upstream_drift_tools.ui.tools_sidebar.jupyter_tab.notebook_loader import (  # noqa: E402
    NotebookLoadError,
    load_notebook,
)
from upstream_drift_tools.ui.tools_sidebar.jupyter_tab.notebook_model import (  # noqa: E402
    CodeCell,
    MarkdownCell,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample.ipynb"


def test_load_notebook_parses_markdown_and_code_cells() -> None:
    doc = load_notebook(FIXTURE)
    assert len(doc.cells) == 2
    md, code = doc.cells
    assert isinstance(md, MarkdownCell)
    assert "Hello world" in md.source
    assert isinstance(code, CodeCell)
    assert "print(" in code.source and "hi" in code.source
    assert len(code.outputs) == 1
    assert code.outputs[0].text == "hi\n"


def test_load_notebook_rejects_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.ipynb"
    with pytest.raises(NotebookLoadError):
        load_notebook(missing)


def test_load_notebook_rejects_malformed_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.ipynb"
    bad.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(NotebookLoadError):
        load_notebook(bad)


def test_load_notebook_rejects_none() -> None:
    with pytest.raises(ValueError):
        load_notebook(None)  # type: ignore[arg-type]
