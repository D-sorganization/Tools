"""Launcher script for PDF Renamer GUI."""

import logging
import sys
from pathlib import Path

# Bootstrap imports for development mode (before pip install -e .)
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src" / "shared" / "python"))
from upstream_drift_tools.bootstrap import ensure_paths  # noqa: E402

ensure_paths(_REPO_ROOT)
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import shared logging configuration
from utils.logging_utils import DEFAULT_FORMAT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=DEFAULT_FORMAT,
    handlers=[
        logging.FileHandler("pdf_renamer.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

if __name__ == "__main__":
    from pdf_renamer.gui import main

    main()
