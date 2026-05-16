"""Sidekick Jupyter notebook tab — Phase 1 read-only viewer (Tools #2875).

Phase 1 scope: render an ``.ipynb`` notebook inside the Sidekick tab as
a read-only document with markdown, code, and raw cells. Execution
(Phase 2 — #2876) and persistence (Phase 3 — #2877) come later.
"""

from .availability import JupyterTabAvailability
from .notebook_loader import NotebookLoadError, load_notebook
from .notebook_model import (
    CellOutput,
    CodeCell,
    MarkdownCell,
    NotebookCell,
    NotebookDocument,
    RawCell,
)
from .unavailable_widget import JupyterUnavailableWidget
from .widget import JupyterNotebookWidget

JUPYTER_TAB_ID = "jupyter"

__all__ = [
    "JUPYTER_TAB_ID",
    "CellOutput",
    "CodeCell",
    "JupyterNotebookWidget",
    "JupyterTabAvailability",
    "JupyterUnavailableWidget",
    "MarkdownCell",
    "NotebookCell",
    "NotebookDocument",
    "NotebookLoadError",
    "RawCell",
    "load_notebook",
]
