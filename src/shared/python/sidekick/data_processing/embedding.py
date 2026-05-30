"""Bridge to the full Data Processor application (single source of truth).

The full-featured Data Processor application lives at
``src/data_processing/data_processor/python/data_processor`` in the Tools repo.
Unlike every other tool (which sits directly under ``src/<tool>`` and is therefore
importable once ``src`` is on ``sys.path``), the Data Processor package is nested
one level deeper, so ``import data_processor`` does not resolve out of the box.

This module is the *single* place that knows how to put that package on the import
path and construct its embeddable widget. Sidekick, Gasification_Model, and
UpstreamDrift all consume the widget through :func:`create_full_data_processor_widget`
so there is exactly one implementation of the real Data Processor UI.

Related issues: D-sorganization/Tools#3111, #3112, #3113.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QWidget

__all__ = [
    "REQUIRED_SUBPATHS",
    "DataProcessorUnavailableError",
    "create_full_data_processor_widget",
    "ensure_full_data_processor_on_path",
    "full_data_processor_available",
]

_logger = logging.getLogger(__name__)

#: Repo-root-relative directories that must be importable for the full app:
#: the Data Processor package root and the shared ``utils`` package root used by
#: ``high_performance_loader``.
REQUIRED_SUBPATHS: tuple[str, ...] = (
    "src/data_processing/data_processor/python",
    "src/python/src",
)

#: Module that exposes the canonical embeddable widget class.
_WIDGET_MODULE = "data_processor.pyqt_widget"
_WIDGET_CLASS = "DataProcessorWidget"


class DataProcessorUnavailableError(RuntimeError):
    """Raised when the full Data Processor application cannot be located/imported."""


def _find_repo_root(start: Path) -> Path:
    """Return the repository root containing the full Data Processor package.

    **Pre-conditions** (DbC): ``start`` must be a :class:`~pathlib.Path`.

    The root is identified by containing both ``pyproject.toml`` and the nested
    Data Processor package directory, so this works from any nesting depth.
    """
    if not isinstance(start, Path):
        raise TypeError(f"start must be a Path, got {type(start)!r}")
    marker = REQUIRED_SUBPATHS[0]
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / marker).is_dir():
            return candidate
    raise DataProcessorUnavailableError(
        "could not locate repository root containing the Data Processor package "
        f"(looked for 'pyproject.toml' and '{marker}' above {start})"
    )


def _evict_shadow_data_processor(app_package_dir: Path) -> None:
    """Drop cached ``data_processor`` modules that resolve outside the full app.

    A small Rust-I/O wrapper at ``src/shared/python/data_processor`` (issue #2989)
    shares the top-level name ``data_processor`` and sits on the always-present
    ``src/shared/python`` path. If it (or any submodule) was imported first it
    would shadow the full application. Evicting cached entries whose file lives
    outside the app package directory lets the next import resolve to the app.
    Already-bound references in other modules are unaffected.
    """
    app_dir = str(app_package_dir.resolve())
    for name in [
        n
        for n in sys.modules
        if n == "data_processor" or n.startswith("data_processor.")
    ]:
        module = sys.modules.get(name)
        module_file = getattr(module, "__file__", None) or ""
        if not module_file or not module_file.startswith(app_dir):
            del sys.modules[name]


def ensure_full_data_processor_on_path(repo_root: Path | None = None) -> Path:
    """Add the full Data Processor package roots to ``sys.path`` (idempotent).

    **Pre-conditions** (DbC): ``repo_root`` is ``None`` or a :class:`~pathlib.Path`.

    The app package directory is inserted ahead of ``src/shared/python`` and any
    shadowing ``data_processor`` modules are evicted, so bare ``import
    data_processor`` resolves to the full application in this process.

    Args:
        repo_root: Optional explicit repository root. When omitted it is located
            relative to this module's file.

    Returns:
        The resolved repository root.
    """
    if repo_root is not None and not isinstance(repo_root, Path):
        raise TypeError(f"repo_root must be a Path or None, got {type(repo_root)!r}")
    root = (
        repo_root
        if repo_root is not None
        else _find_repo_root(Path(__file__).resolve())
    )
    # Insert in reverse so REQUIRED_SUBPATHS[0] (the app package dir) lands first.
    # Force it ahead of any existing occurrence — the app dir is often already on
    # sys.path *behind* ``src/shared/python``, which would let the issue-#2989
    # ``data_processor`` wrapper shadow the application.
    for sub in reversed(REQUIRED_SUBPATHS):
        target = root / sub
        if not target.is_dir():
            continue
        path_str = str(target)
        while path_str in sys.path:
            sys.path.remove(path_str)
        sys.path.insert(0, path_str)
    _evict_shadow_data_processor(root / REQUIRED_SUBPATHS[0])
    return root


def full_data_processor_available() -> bool:
    """Return ``True`` if the full Data Processor widget can be imported."""
    try:
        ensure_full_data_processor_on_path()
        return importlib.util.find_spec(_WIDGET_MODULE) is not None
    except (DataProcessorUnavailableError, ImportError, ValueError) as exc:
        _logger.debug("Full Data Processor not available: %s", exc)
        return False


def create_full_data_processor_widget(parent: QWidget | None = None) -> QWidget:
    """Construct the full Data Processor embeddable widget (single source of truth).

    Args:
        parent: Optional Qt parent widget.

    Returns:
        The real :class:`data_processor.pyqt_widget.DataProcessorWidget` instance.

    Raises:
        DataProcessorUnavailableError: If the package or its dependencies cannot
            be imported.
    """
    ensure_full_data_processor_on_path()
    try:
        module = importlib.import_module(_WIDGET_MODULE)
        widget_class = getattr(module, _WIDGET_CLASS)
    except ImportError as exc:
        raise DataProcessorUnavailableError(
            f"failed to import {_WIDGET_MODULE}.{_WIDGET_CLASS}: {exc}"
        ) from exc
    return cast("QWidget", widget_class(parent))
