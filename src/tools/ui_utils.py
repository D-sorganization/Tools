"""Shared UI utilities for Tools applications."""

import logging
from pathlib import Path
from typing import Any

try:
    from upstream_drift_tools.utils.paths import get_repo_root
except ImportError:
    try:
        from tools.launch_utils import get_repo_root
    except ImportError:

        def get_repo_root() -> Path:  # type: ignore[misc]
            return Path(__file__).resolve().parent.parent


logger = logging.getLogger(__name__)


def find_icon(name: str = "tools_icon.ico") -> Path | None:
    """Find a UI icon by searching standard locations."""
    repo_root: Path = get_repo_root()

    candidates: list[Path] = [
        repo_root / name,
        repo_root / "resources" / name,
        repo_root / "tools" / "gui" / "resources" / name,
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def set_tk_icon(root: Any, icon_name: str = "tools_icon.ico") -> bool:
    """Set process icon for a Tkinter root/toplevel window.

    Args:
        root: Tkinter root or Toplevel instance.
        icon_name: Name of the icon file.

    Returns:
        True if icon was set, False otherwise.
    """
    icon_path = find_icon(icon_name)
    if not icon_path:
        return False

    try:
        # iconbitmap is Windows specific mostly, but handles .ico well
        root.iconbitmap(str(icon_path))
        return True
    except Exception as e:
        logger.warning(f"Could not set Tk icon {icon_path}: {e}")
        return False


def set_qt_icon(window: Any, icon_name: str = "tools_icon.ico") -> bool:
    """Set process icon for a PyQt window.

    Args:
        window: PyQt QWidget/QMainWindow instance.
        icon_name: Name of the icon file.

    Returns:
        True if icon was set, False otherwise.
    """
    icon_path = find_icon(icon_name)
    if not icon_path:
        return False

    try:
        from PyQt6.QtGui import QIcon

        window.setWindowIcon(QIcon(str(icon_path)))
        return True
    except ImportError:
        logger.warning("PyQt6 not installed, cannot set Qt icon")
        return False
    except Exception as e:
        logger.warning(f"Could not set Qt icon {icon_path}: {e}")
        return False
