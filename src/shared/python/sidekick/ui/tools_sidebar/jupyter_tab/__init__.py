"""Sidekick Jupyter notebook tab — Phase 1 read-only viewer + Phase 2 session model.

Phase 1 (Tools #2875): render an ``.ipynb`` notebook inside the Sidekick tab
as a read-only document with markdown, code, and raw cells.

Phase 2 (Tools #2876): session model with path validation and secure
persistence via :class:`NotebookSessionModel` and
:class:`NotebookSessionManager`.  :class:`SidekickNotebookWidget` is the
session-aware widget wrapper introduced in Phase 2.
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
from .notebook_session import NotebookSessionManager, NotebookSessionModel
from .sidekick_notebook_widget import SidekickNotebookWidget
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
    "NotebookSessionManager",
    "NotebookSessionModel",
    "RawCell",
    "SidekickNotebookWidget",
    "load_notebook",
]
