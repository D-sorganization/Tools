"""Sidekick Jupyter Notebook tab — Phase 1.

Issue #2875: [Jupyter Sidekick Phase 1] Notebook UI Tab and Dependency Management.

Provides :func:`build_notebook_tab` (the Sidekick factory) and
:class:`SidekickNotebookWidget` (the underlying QWidget).

When ``jupyter_client`` and ``nbformat`` are not installed the widget degrades
gracefully: it shows an actionable install prompt instead of crashing.
When they are installed it renders a Phase-1 placeholder ready for Phase 2/3
kernel integration.

Design-by-contract: public methods validate inputs and raise ``TypeError`` for
wrong types, following the project-wide DbC convention.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .help_content import DEFAULT_SIDEBAR_TAB_HELP
from .qt_compat import QtWidgets

logger = logging.getLogger(__name__)

NOTEBOOK_TAB_ID = "notebook"

_JUPYTER_INSTALL_HINT = (
    "Jupyter is not installed.\n"
    "Run: pip install jupyter_client nbformat\n"
    "or: pip install jupyter"
)

# ---------------------------------------------------------------------------
# Public factory (matches the Sidekick tab factory signature)
# ---------------------------------------------------------------------------


def build_notebook_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build the Jupyter Notebook tab for a Sidekick sidebar.

    Returns a :class:`SidekickNotebookWidget` when the optional Jupyter
    dependencies are available, or a lightweight install-prompt widget when
    they are absent.

    Args:
        sidebar: The host :class:`UnifiedToolsSidebar` instance (or any object
            exposing ``project_root``).

    Returns:
        A :class:`QtWidgets.QWidget` suitable for embedding in the sidebar tab strip.
    """
    project_root = getattr(sidebar, "project_root", Path("."))
    widget = SidekickNotebookWidget(project_root=project_root, parent=sidebar)
    tooltip = DEFAULT_SIDEBAR_TAB_HELP.get(NOTEBOOK_TAB_ID, {}).get("summary", "")
    if tooltip:
        widget.setToolTip(tooltip)
    return widget


# ---------------------------------------------------------------------------
# Widget implementation
# ---------------------------------------------------------------------------


class SidekickNotebookWidget(QtWidgets.QWidget):
    """Sidekick tab widget for Jupyter notebook interaction.

    Phase 1 scope:
    - Graceful degradation when ``jupyter_client`` / ``nbformat`` are absent.
    - Session metadata (``notebook_path``, ``kernel_env``) isolated per instance.
    - Actionable install prompt surfaced as a ``QLabel`` when Jupyter is missing.
    - Stub ready for Phase 2 kernel connection and notebook rendering.
    """

    OBJECT_NAME = "SidekickNotebookTab"

    def __init__(
        self,
        *,
        project_root: Path,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        if project_root is None:
            raise ValueError("project_root must be provided")
        super().__init__(parent)
        self.setObjectName(self.OBJECT_NAME)
        self._project_root = project_root
        # Session metadata — independent per instance (no class-level mutable state).
        self.session_metadata: dict[str, str | None] = {
            "notebook_path": None,
            "kernel_env": None,
        }
        self._jupyter_available = _check_jupyter_available()
        self._build_ui()

    # ------------------------------------------------------------------
    # Public session-management API
    # ------------------------------------------------------------------

    def open_notebook(self, path: str) -> None:
        """Record the notebook file path in session metadata.

        Args:
            path: Absolute or project-relative path to the ``.ipynb`` file.

        Raises:
            TypeError: If ``path`` is not a :class:`str`.
        """
        if not isinstance(path, str):
            raise TypeError(f"path must be a str, got {type(path).__name__!r}")
        self.session_metadata["notebook_path"] = path
        logger.debug("NotebookTab: notebook_path set to %r", path)

    def set_kernel_environment(self, env: str) -> None:
        """Record the kernel environment name in session metadata.

        Args:
            env: Name of the Python environment or kernel spec (e.g. ``"my-venv"``).

        Raises:
            TypeError: If ``env`` is not a :class:`str`.
        """
        if not isinstance(env, str):
            raise TypeError(f"env must be a str, got {type(env).__name__!r}")
        self.session_metadata["kernel_env"] = env
        logger.debug("NotebookTab: kernel_env set to %r", env)

    # ------------------------------------------------------------------
    # Internal UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        if self._jupyter_available:
            self._build_available_ui(layout)
        else:
            self._build_install_prompt_ui(layout)

    def _build_available_ui(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Phase-1 placeholder shown when Jupyter deps are present."""
        heading = QtWidgets.QLabel("Jupyter Notebook", self)
        heading.setObjectName("SidekickNotebookHeading")
        heading.setWordWrap(True)
        heading.setToolTip("Jupyter Notebook tab — Phase 1 stub.")
        layout.addWidget(heading)

        info = QtWidgets.QLabel(
            "Jupyter is installed. Notebook kernel integration coming in Phase 2.",
            self,
        )
        info.setObjectName("SidekickNotebookInfo")
        info.setWordWrap(True)
        layout.addWidget(info)
        layout.addStretch(1)

    def _build_install_prompt_ui(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Actionable install prompt shown when Jupyter deps are absent."""
        heading = QtWidgets.QLabel("Jupyter Notebook", self)
        heading.setObjectName("SidekickNotebookHeading")
        heading.setWordWrap(True)
        heading.setToolTip("Jupyter Notebook tab — dependencies not installed.")
        layout.addWidget(heading)

        message = QtWidgets.QLabel(_JUPYTER_INSTALL_HINT, self)
        message.setObjectName("SidekickNotebookInstallHint")
        message.setWordWrap(True)
        message.setToolTip("Run this command to enable the Jupyter Notebook tab.")
        layout.addWidget(message)

        copy_btn = QtWidgets.QPushButton("Copy install command", self)
        copy_btn.setObjectName("SidekickNotebookCopyInstall")
        copy_btn.setToolTip("Copy the pip install command to the clipboard.")
        copy_btn.clicked.connect(self._copy_install_command)
        layout.addWidget(copy_btn)

        layout.addStretch(1)

    def _copy_install_command(self) -> None:
        """Copy the pip install command to the system clipboard."""
        clipboard = QtWidgets.QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText("pip install jupyter_client nbformat")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _check_jupyter_available() -> bool:
    """Return True when both ``jupyter_client`` and ``nbformat`` are importable.

    When either package is blocked via ``sys.modules[name] = None`` (the standard
    monkeypatching technique) ``importlib.import_module`` raises ``ImportError``,
    so the ``except`` branch handles that case correctly.
    """
    import importlib

    for pkg in ("jupyter_client", "nbformat"):
        try:
            importlib.import_module(pkg)
        except ImportError:
            return False
    return True
