"""Shared notes workspace package.

Provides a reusable project-backed notes file model with safe deletion
(recycle bin) and a PyQt dock widget that can be embedded or popped out.
"""

from __future__ import annotations

from .models import RecycledNoteItem
from .storage import NotesStorage

try:
    from .integration import attach_notes_dock
    from .notes_dock_widget import NotesDockWidget

    _PYQT6_AVAILABLE = True
except ImportError:
    _PYQT6_AVAILABLE = False
    NotesDockWidget = None  # type: ignore[assignment, misc]
    attach_notes_dock = None  # type: ignore[assignment, misc]

__all__ = [
    "NotesStorage",
    "RecycledNoteItem",
    "NotesDockWidget",
    "attach_notes_dock",
]
