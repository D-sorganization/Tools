"""SidekickNotebookWidget — Phase 2 session-aware notebook wrapper.

Wraps :class:`~.notebook_session.NotebookSessionModel` so that the Sidekick
tab always validates notebook paths before accepting them and exposes a clean
session API for Phase 2 (Tools #2876).

This module deliberately avoids Qt imports so that it can be unit-tested in a
headless environment.  The actual Qt rendering is delegated to
:class:`~.widget.JupyterNotebookWidget`; this class owns only the *session*
state.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .notebook_session import NotebookSessionModel

logger = logging.getLogger(__name__)


class SidekickNotebookWidget:
    """Session-aware wrapper for a Sidekick Jupyter notebook tab.

    Owns a :class:`~.notebook_session.NotebookSessionModel` that tracks
    which notebook is open, the active virtual environment, and when the
    session was last persisted.

    The widget does NOT hold the full notebook JSON — it holds only a path
    reference validated against ``_workspace_root``.

    Typical usage::

        widget = SidekickNotebookWidget(workspace_root=Path("/project"))
        widget.open_notebook("/project/notebooks/analysis.ipynb")
        widget.set_kernel_environment("venv311")
        meta = widget.session_metadata  # dict for UI display

    Attributes:
        _workspace_root: Directory that all notebook paths must be contained
            within.  Set by the constructor (or directly by tests via
            ``__new__``).
        _session: The current :class:`~.notebook_session.NotebookSessionModel`,
            or ``None`` if no notebook has been opened yet.
    """

    def __init__(self, workspace_root: Path | str) -> None:
        self._workspace_root: Path = Path(workspace_root)
        self._session: NotebookSessionModel | None = None

    # ------------------------------------------------------------------
    # Public API (Phase 1 compat + Phase 2 extensions)
    # ------------------------------------------------------------------

    def open_notebook(self, path: str | Path) -> None:
        """Open *path* as the active notebook for this session.

        Validates that *path* resolves inside ``_workspace_root`` before
        storing it.  This is the primary DbC guard against directory-traversal
        attacks.

        Args:
            path: Absolute or relative path to the ``.ipynb`` file.  Relative
                paths are resolved against the current working directory, then
                re-validated against ``_workspace_root``.

        Raises:
            ValueError: If *path* resolves outside ``_workspace_root``.
        """
        notebook_path = Path(path)
        model = NotebookSessionModel(
            notebook_path=notebook_path,
            workspace_root=self._workspace_root,
            kernel_env=None,
        )
        model.validate_path()  # DbC — raises ValueError on traversal
        self._session = model
        logger.debug("Notebook session opened: %s", notebook_path)

    def set_kernel_environment(self, env: str | None) -> None:
        """Set the virtual-environment name for the active kernel.

        Args:
            env: Virtual-environment name (e.g. ``"venv311"``), or ``None``
                to clear the selection.

        Raises:
            RuntimeError: If no notebook has been opened yet.
        """
        if self._session is None:
            raise RuntimeError(
                "Cannot set kernel environment: no notebook is open. "
                "Call open_notebook() first."
            )
        self._session.kernel_env = env

    @property
    def session_metadata(self) -> dict[str, object]:
        """Return a plain-dict snapshot of the current session state.

        Provides a stable Phase-1-compatible surface for UI components that
        expect a ``dict``.  The full notebook JSON is never included.

        Returns:
            A dict with keys ``notebook_path``, ``kernel_env``, and
            ``last_saved``.  Values are ``None`` when no notebook is open.
        """
        if self._session is None:
            return {
                "notebook_path": None,
                "kernel_env": None,
                "last_saved": None,
            }
        return {
            "notebook_path": str(self._session.notebook_path),
            "kernel_env": self._session.kernel_env,
            "last_saved": (
                self._session.last_saved.isoformat()
                if self._session.last_saved is not None
                else None
            ),
        }
