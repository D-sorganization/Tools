"""Path setup for urdf_viewer tests."""

import sys
from pathlib import Path

# Add repo root so src.shared.python imports work
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Add shared python dir for cors module direct imports
_SHARED_DIR = str(
    Path(__file__).resolve().parent.parent.parent.parent / "shared" / "python"
)
if _SHARED_DIR not in sys.path:
    sys.path.insert(0, _SHARED_DIR)

# Add urdf_builder_gui python dir
_URDF_BUILDER_DIR = str(
    Path(__file__).resolve().parent.parent.parent.parent / "urdf_builder_gui" / "python"
)
if _URDF_BUILDER_DIR not in sys.path:
    sys.path.insert(0, _URDF_BUILDER_DIR)

# Add the app dir itself (so app.py is importable)
_APP_DIR = str(Path(__file__).resolve().parent.parent)
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)
