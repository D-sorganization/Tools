"""Launcher script for PDF Renamer GUI."""

import logging
import sys

from _bootstrap import bootstrap  # noqa: E402

_REPO_ROOT = bootstrap(__file__)

# Import shared logging configuration
from utils.logging_utils import DEFAULT_FORMAT  # noqa: E402

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
