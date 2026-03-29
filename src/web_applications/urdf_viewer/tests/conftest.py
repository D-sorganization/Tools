"""Path setup for urdf_viewer tests.

Consolidates all path setup into a single helper function to avoid
scattered sys.path.insert calls.
"""

import sys
from pathlib import Path


def _setup_urdf_viewer_paths() -> None:
    """Add all required directories for urdf_viewer testing."""
    base = Path(__file__).resolve().parent.parent
    src_dir = base.parent.parent
    paths = [
        str(src_dir.parent),  # repo root
        str(src_dir / "shared" / "python"),  # cors, theme
        str(src_dir / "urdf_builder_gui" / "python"),  # urdf_builder_gui
        str(base),  # app dir
    ]
    for p in paths:
        if p not in sys.path:
            sys.path.insert(0, p)


_setup_urdf_viewer_paths()
